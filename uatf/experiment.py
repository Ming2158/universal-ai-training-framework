"""Dataset construction and shared training experiment execution."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import logging
import platform
import random
import shutil
import subprocess
import time
import uuid
from collections.abc import Callable, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import yaml
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, Subset

from uatf import __version__
from uatf.config import dump_config
from uatf.data import (
    CsvImageClassificationDataset,
    CsvMultiLabelDataset,
    ImageFolderDataset,
    build_image_transform,
    collate_image_batch,
    create_data_loaders,
)
from uatf.evaluation import evaluate_training_run
from uatf.experiments import register_model_asset
from uatf.logging_utils import configure_logging, shutdown_logging
from uatf.models import build_model_from_config, get_loss, get_model_adapter
from uatf.training import (
    Trainer,
    TrainingEvent,
    build_early_stopping,
    build_optimizer,
    build_scheduler,
)
from uatf.training.trainer import TaskStatusCallback, TrainingEventEmitter

logger = logging.getLogger("uatf.cli")


def _path(value: str | None) -> Path | None:
    return Path(value).expanduser().resolve() if value is not None else None


def _image_loaders(
    configuration: Mapping[str, Any], task: str, seed: int
) -> tuple[DataLoader, DataLoader, list[str], dict[str, Any]]:
    dataset_config = configuration["dataset"]
    training_config = configuration["training"]
    resize_enabled = dataset_config.get("resize_enabled", True)
    transform = build_image_transform(
        image_size=dataset_config["image_size"],
        mean=dataset_config["mean"],
        std=dataset_config["std"],
        resize_enabled=resize_enabled,
    )
    training_transform = build_image_transform(
        image_size=dataset_config["image_size"],
        mean=dataset_config["mean"],
        std=dataset_config["std"],
        augmentation=dataset_config["augmentation"],
        training=True,
        resize_enabled=resize_enabled,
    )
    batch_size = training_config["batch_size"]
    num_workers = training_config["num_workers"]
    pin_memory = training_config["device"] != "cpu" and torch.cuda.is_available()
    data_format = dataset_config["format"]

    if data_format == "image_folder":
        train_path = _path(dataset_config.get("train_dir"))
        validation_path = _path(dataset_config.get("validation_dir"))
        root = _path(dataset_config.get("path"))
        if train_path is None and root is not None:
            if (root / "train").is_dir() and (root / "validation").is_dir():
                train_path, validation_path = root / "train", root / "validation"
            elif (root / "train").is_dir() and (root / "val").is_dir():
                train_path, validation_path = root / "train", root / "val"
            elif (root / "train").is_dir():
                train_path = root / "train"
            else:
                train_path = root

        if train_path is None:
            raise ValueError("Image-folder data requires dataset.path or dataset.train_dir.")
        train_dataset = ImageFolderDataset(
            train_path, transform=training_transform, verify_images=False
        )
        if validation_path is not None:
            validation_dataset = ImageFolderDataset(
                validation_path, transform=transform, verify_images=False
            )
            if train_dataset.classes != validation_dataset.classes:
                raise ValueError(
                    "Training and validation folders must contain the same class directories "
                    "in the same sorted order."
                )
            train_loader = _loader(
                train_dataset,
                batch_size,
                num_workers,
                shuffle=True,
                seed=seed,
                pad_variable_images=not resize_enabled,
                pin_memory=pin_memory,
            )
            validation_loader = _loader(
                validation_dataset,
                batch_size,
                num_workers,
                shuffle=False,
                seed=seed,
                pad_variable_images=not resize_enabled,
                pin_memory=pin_memory,
            )
            validation_count = len(validation_dataset)
            validation_source = str(validation_path)
        else:
            train_loader, optional_validation_loader, _ = create_data_loaders(
                train_dataset,
                batch_size=batch_size,
                validation_fraction=dataset_config["validation_fraction"],
                seed=seed,
                num_workers=num_workers,
                train_transform=training_transform,
                evaluation_transform=transform,
                pad_variable_images=not resize_enabled,
                pin_memory=pin_memory,
            )
            if optional_validation_loader is None:
                raise ValueError("A validation dataset is required for evaluation.")
            validation_loader = optional_validation_loader
            validation_count = len(validation_loader.dataset)
            validation_source = f"random split ({dataset_config['validation_fraction']:.0%})"

        classes = list(train_dataset.classes)
        metadata = {
            "format": data_format,
            "path": str(train_path),
            "validation_source": validation_source,
            "validation_path": str(validation_path) if validation_path else None,
            "train_samples": len(train_loader.dataset),
            "validation_samples": validation_count,
            "class_names": classes,
        }
        return train_loader, validation_loader, classes, metadata

    annotation_path = _path(dataset_config.get("annotations_csv"))
    configured_path = _path(dataset_config.get("path"))
    if annotation_path is None and configured_path is not None:
        if configured_path.is_file():
            annotation_path = configured_path
        elif configured_path.is_dir():
            annotation_path = configured_path / "annotations.csv"
    if annotation_path is None:
        raise ValueError("CSV data requires dataset.annotations_csv or dataset.path.")

    image_root = _path(dataset_config.get("image_dir"))
    if image_root is None:
        if configured_path is not None and configured_path.is_dir():
            image_candidate = configured_path / "images"
            image_root = image_candidate if image_candidate.is_dir() else configured_path
        else:
            image_root = annotation_path.parent / "images"
            if not image_root.is_dir():
                image_root = annotation_path.parent

    labels = dataset_config.get("label_columns")
    if labels is not None and (
        not isinstance(labels, list)
        or any(not isinstance(label, str) or not label for label in labels)
    ):
        raise ValueError("dataset.label_columns must be a list of non-empty column names.")
    if data_format == "csv_multilabel":
        full_dataset: Dataset = CsvMultiLabelDataset(
            annotation_path,
            image_root=image_root,
            transform=transform,
            label_columns=labels,
            verify_images=False,
        )
    else:
        if labels is not None:
            raise ValueError("dataset.label_columns is only supported for csv_multilabel.")
        full_dataset = CsvImageClassificationDataset(
            annotation_path,
            image_root=image_root,
            transform=transform,
            verify_images=False,
        )

    train_loader, validation_loader, _ = create_data_loaders(
        full_dataset,
        batch_size=batch_size,
        validation_fraction=dataset_config["validation_fraction"],
        seed=seed,
        num_workers=num_workers,
        train_transform=training_transform,
        evaluation_transform=transform,
        pad_variable_images=not resize_enabled,
        pin_memory=pin_memory,
    )
    if validation_loader is None:
        raise ValueError("A validation dataset is required for evaluation.")
    classes = list(full_dataset.classes)
    metadata = {
        "format": data_format,
        "annotations_csv": str(annotation_path),
        "image_dir": str(image_root),
        "validation_source": f"random split ({dataset_config['validation_fraction']:.0%})",
        "train_samples": len(train_loader.dataset),
        "validation_samples": len(validation_loader.dataset),
        "class_names": classes,
    }
    return train_loader, validation_loader, classes, metadata


def _loader(
    dataset: Dataset,
    batch_size: int,
    num_workers: int,
    *,
    shuffle: bool,
    seed: int,
    pad_variable_images: bool = False,
    pin_memory: bool = False,
) -> DataLoader:
    options: dict[str, Any] = {"num_workers": num_workers, "pin_memory": pin_memory}
    if num_workers > 0:
        options.update(persistent_workers=True, prefetch_factor=2)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=torch.Generator().manual_seed(seed) if shuffle else None,
        collate_fn=collate_image_batch if pad_variable_images else None,
        **options,
    )


def _classification_accuracy(logits: Tensor, targets: Tensor) -> Tensor:
    return (logits.argmax(dim=1) == targets).float().mean()


def _multilabel_micro_f1(threshold: float):
    def metric(logits: Tensor, targets: Tensor) -> Tensor:
        predictions = logits.sigmoid() >= threshold
        true_positive = (predictions & (targets >= 0.5)).sum()
        denominator = predictions.sum() + (targets >= 0.5).sum()
        return 2 * true_positive.float() / denominator.clamp_min(1)

    return metric


def _experiment_directory(output_config: Mapping[str, Any]) -> Path:
    base = Path(output_config["directory"]).expanduser().resolve()
    base.mkdir(parents=True, exist_ok=True)
    requested_name = output_config.get("experiment_name")
    if requested_name:
        destination = base / requested_name
        if destination.exists():
            raise FileExistsError(f"Experiment output already exists: {destination}")
        destination.mkdir()
        return destination

    index = 1
    while True:
        destination = base / f"experiment_{index:03d}"
        try:
            destination.mkdir()
            return destination
        except FileExistsError:
            index += 1


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_name(f"{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _environment_info() -> dict[str, Any]:
    repository = Path(__file__).resolve().parents[1]
    try:
        commit = subprocess.run(
            ["git", "-C", str(repository), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        ).stdout.strip() or None
        dirty = bool(
            subprocess.run(
                ["git", "-C", str(repository), "status", "--porcelain"],
                capture_output=True,
                text=True,
                timeout=2,
                check=False,
            ).stdout.strip()
        )
    except (OSError, subprocess.TimeoutExpired):
        commit, dirty = None, None
    dependencies: dict[str, str | None] = {}
    for distribution, key in (
        ("torchvision", "torchvision"),
        ("Pillow", "pillow"),
        ("PyYAML", "pyyaml"),
        ("gradio", "gradio"),
    ):
        try:
            dependencies[key] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            dependencies[key] = None
    return {
        "uatf": __version__,
        "python": platform.python_version(),
        "pytorch": str(torch.__version__),
        "platform": platform.platform(),
        "git_commit": commit,
        "git_dirty": dirty,
        "dependencies": dependencies,
    }


def _labels_for_target(target: Any, class_names: list[str]) -> list[str] | str:
    if isinstance(target, (tuple, list)):
        return [
            name
            for name, value in zip(class_names, target, strict=True)
            if bool(value)
        ]
    return class_names[int(target)]


def _loader_samples(
    loader: DataLoader, split: str, class_names: list[str]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    dataset = loader.dataset
    indices = dataset.indices if isinstance(dataset, Subset) else range(len(dataset))
    while isinstance(dataset, Subset):
        dataset = dataset.dataset
    samples = getattr(dataset, "samples", None)
    if samples is None:
        raise TypeError("Experiment manifests require image datasets with stable sample paths.")
    root = getattr(dataset, "root", getattr(dataset, "image_root", None))
    root_path = Path(root).resolve() if root is not None else None
    selected: list[dict[str, Any]] = []
    all_samples: list[dict[str, Any]] = []
    selected_indices = set(indices)
    for index, (sample_path, target) in enumerate(samples):
        absolute = Path(sample_path).resolve()
        try:
            sample_id = absolute.relative_to(root_path).as_posix() if root_path else absolute.name
        except ValueError:
            sample_id = absolute.as_posix()
        row = {"sample_id": sample_id, "labels": _labels_for_target(target, class_names)}
        all_samples.append(row)
        if index in selected_indices:
            selected.append({**row, "split": split})
    return all_samples, selected


def _record_dataset_manifest(
    output_directory: Path,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    class_names: list[str],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    train_all, train_rows = _loader_samples(train_loader, "train", class_names)
    validation_all, validation_rows = _loader_samples(
        validation_loader, "validation", class_names
    )
    format_name = str(metadata["format"])
    all_rows = [
        *({"source": "train", **row} for row in train_all),
        *({"source": "validation", **row} for row in validation_all),
    ]
    split_rows = [*train_rows, *validation_rows]
    canonical = json.dumps(
        {"format": format_name, "class_names": class_names, "samples": split_rows},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    dataset_fingerprint = hashlib.sha256(
        json.dumps(
            {"format": format_name, "class_names": class_names, "samples": all_rows},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    split_id = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    manifest = {
        "schema_version": 1,
        "dataset_id": dataset_fingerprint,
        "split_id": split_id,
        "format": format_name,
        "class_names": class_names,
        "samples": split_rows,
    }
    manifest_path = output_directory / ".uatf" / "datasets" / f"{split_id}.json"
    if not manifest_path.is_file():
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        _write_json(manifest_path, manifest)
    return {
        **metadata,
        "id": dataset_fingerprint,
        "split_id": split_id,
        "manifest_path": manifest_path.relative_to(output_directory).as_posix(),
        "sample_count": len(all_rows),
        "train_samples": len(train_rows),
        "validation_samples": len(validation_rows),
        "manifest_sha256": split_id,
    }


def _update_run_record(
    experiment_dir: Path, record: dict[str, Any], started_monotonic: float
) -> None:
    record["duration_seconds"] = round(time.monotonic() - started_monotonic, 3)
    _write_json(experiment_dir / "experiment.json", record)


def _register_experiment_model_asset(
    experiment_dir: Path,
    record: dict[str, Any],
    configuration: Mapping[str, Any],
    evaluation: Mapping[str, Any] | None,
) -> dict[str, Any]:
    model_config = configuration["model"]
    training_config = configuration["training"]
    dataset_config = configuration["dataset"]
    return register_model_asset(
        experiment_dir.parent,
        {
            "id": f"model-{record['id']}",
            "name": experiment_dir.name,
            "architecture": model_config["name"],
            "pretrained": model_config["pretrained"],
            "experiment_id": record["id"],
            "experiment_directory": experiment_dir.name,
            "weights_path": experiment_dir / "best_model.pth",
            "configuration_path": experiment_dir / "config.yaml",
            "task": configuration["task"]["type"],
            "dataset_format": dataset_config["format"],
            "class_names": list(record.get("dataset", {}).get("class_names", [])),
            "preprocessing": {
                key: dataset_config.get(key)
                for key in ("image_size", "resize_enabled", "mean", "std")
            },
            "training_summary": {
                "epochs_completed": record.get("training", {}).get("epochs_completed", 0),
                "batch_size": training_config["batch_size"],
                "learning_rate": training_config["learning_rate"],
                "seed": training_config["seed"],
            },
            "evaluation": dict(evaluation or {}),
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "version": 1,
            "weights_kind": "full_training_checkpoint",
        },
    )


def run_experiment(
    configuration: dict[str, Any],
    source_config: Path | None,
    *,
    source_configuration: Mapping[str, Any] | None = None,
    configuration_overrides: Mapping[str, Any] | None = None,
    event_callback: Callable[[TrainingEvent], None] | None = None,
    stop_requested: Callable[[], bool] | None = None,
    task_status_callback: TaskStatusCallback | None = None,
    task_directory_callback: Callable[[Path], None] | None = None,
) -> Path:
    """Create a stable run record, then execute the shared training workflow."""
    experiment_dir = _experiment_directory(configuration["output"])
    experiment_id = uuid.uuid4().hex
    started_monotonic = time.monotonic()
    created_at = datetime.now(timezone.utc).isoformat()
    experiment_dir.joinpath("logs").mkdir()
    dump_config(source_configuration or configuration, experiment_dir / "source_config.yaml")
    dump_config(configuration, experiment_dir / "config.yaml")
    record: dict[str, Any] = {
        "schema_version": 1,
        "id": experiment_id,
        "name": experiment_dir.name,
        "directory": experiment_dir.name,
        "created_at_utc": created_at,
        "started_at_utc": created_at,
        "finished_at_utc": None,
        "status": "Preparing",
        "task": configuration["task"]["type"],
        "dataset": {"format": configuration["dataset"]["format"]},
        "model": {
            "name": configuration["model"]["name"],
            "pretrained": configuration["model"]["pretrained"],
            "num_classes": configuration["model"].get("num_classes"),
        },
        "configuration": {
            "source": "source_config.yaml",
            "effective": "config.yaml",
            "source_path": str(source_config) if source_config is not None else None,
            "overrides": dict(configuration_overrides or {}),
        },
        "code": _environment_info(),
        "environment": {"requested_device": configuration["training"]["device"]},
        "training": {"requested_epochs": configuration["training"]["epochs"]},
        "evaluation": {},
        "artifacts": {
            "log": "logs/train.log",
            "metrics": "metrics.json",
            "source_config": "source_config.yaml",
            "effective_config": "config.yaml",
        },
        "failure": None,
    }
    _update_run_record(experiment_dir, record, started_monotonic)
    try:
        if task_directory_callback is not None:
            task_directory_callback(experiment_dir)
        result = _execute_experiment(
            configuration,
            source_config,
            source_configuration=source_configuration,
            configuration_overrides=configuration_overrides,
            event_callback=event_callback,
            stop_requested=stop_requested,
            task_status_callback=task_status_callback,
            experiment_dir=experiment_dir,
            experiment_record=record,
            started_monotonic=started_monotonic,
        )
        if record["status"] == "Preparing":
            record["status"] = "Completed"
        if record["status"] in {"Completed", "Stopped"}:
            record["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        _update_run_record(experiment_dir, record, started_monotonic)
        return result
    except Exception as error:
        record["status"] = "Failed"
        record["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        record["failure"] = {"type": type(error).__name__, "message": str(error)}
        try:
            if any(
                getattr(handler, "_uatf_managed_handler", False)
                for handler in logging.getLogger("uatf").handlers
            ):
                logger.exception("Experiment %s failed", experiment_id)
                shutdown_logging()
        except Exception:
            pass
        _update_run_record(experiment_dir, record, started_monotonic)
        if event_callback is not None:
            try:
                TrainingEventEmitter(event_callback)(
                    TrainingEvent(
                        event_type="training_failed",
                        experiment_dir=str(experiment_dir),
                        message=str(error),
                    )
                )
            except Exception:
                logger.debug("Could not emit the experiment failure event", exc_info=True)
        raise


def _sync_optimizer_config(
    configuration: dict[str, Any], trainer: Trainer
) -> None:
    training_config = configuration["training"]
    parameter_group = trainer.optimizer.param_groups[0]
    training_config["learning_rate"] = float(parameter_group["lr"])
    optimizer_config = training_config["optimizer"]
    optimizer_config["weight_decay"] = float(parameter_group["weight_decay"])
    if isinstance(trainer.optimizer, (torch.optim.Adam, torch.optim.AdamW)):
        optimizer_config["betas"] = [float(value) for value in parameter_group["betas"]]
        optimizer_config["eps"] = float(parameter_group["eps"])
    elif isinstance(trainer.optimizer, torch.optim.SGD):
        optimizer_config["momentum"] = float(parameter_group["momentum"])
        optimizer_config["nesterov"] = bool(parameter_group["nesterov"])
    if trainer.scheduler is not None:
        scheduler_config = training_config["scheduler"]
        scheduler_config["enabled"] = True
        if isinstance(trainer.scheduler, torch.optim.lr_scheduler.StepLR):
            scheduler_config.update(
                {
                    "name": "step_lr",
                    "step_size": int(trainer.scheduler.step_size),
                    "gamma": float(trainer.scheduler.gamma),
                }
            )
        elif isinstance(trainer.scheduler, torch.optim.lr_scheduler.CosineAnnealingLR):
            scheduler_config.update(
                {
                    "name": "cosine",
                    "t_max": int(trainer.scheduler.T_max),
                    "eta_min": float(trainer.scheduler.eta_min),
                }
            )


def _preserve_best_checkpoint(resume_from: str, experiment_dir: Path) -> None:
    source = Path(resume_from).expanduser()
    previous_best = source.parent / "best_model.pth"
    checkpoint = previous_best if previous_best.is_file() else source
    destination = experiment_dir / "best_model.pth"
    if checkpoint.resolve() != destination.resolve():
        shutil.copy2(checkpoint, destination)


def _execute_experiment(
    configuration: dict[str, Any],
    source_config: Path | None,
    *,
    source_configuration: Mapping[str, Any] | None = None,
    configuration_overrides: Mapping[str, Any] | None = None,
    event_callback: Callable[[TrainingEvent], None] | None = None,
    stop_requested: Callable[[], bool] | None = None,
    task_status_callback: TaskStatusCallback | None = None,
    experiment_dir: Path,
    experiment_record: dict[str, Any],
    started_monotonic: float,
) -> Path:
    """Execute one run inside the directory allocated by :func:`run_experiment`."""
    task = configuration["task"]["type"]
    training_config = configuration["training"]
    event_emitter = TrainingEventEmitter(event_callback)
    if event_callback is not None:
        event_emitter(
            TrainingEvent(
                event_type="training_started",
                experiment_dir=str(experiment_dir),
                total_epochs=training_config["epochs"],
            )
        )
    model_adapter = get_model_adapter(configuration["model"]["name"])
    seed = training_config["seed"]
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    log_path = experiment_dir / "logs" / "train.log"
    configure_logging(log_path)
    started_at = experiment_record["started_at_utc"]
    if training_config["device"] == "cuda" and not torch.cuda.is_available():
        raise ValueError(
            "CUDA was requested but is unavailable in this PyTorch environment; "
            "set training.device to cpu or install CUDA-enabled PyTorch."
        )

    train_loader, validation_loader, class_names, dataset_metadata = _image_loaders(
        configuration, task, seed
    )
    model_config = configuration["model"]
    configured_count = model_config.get("num_classes")
    if configured_count is not None and configured_count != len(class_names):
        raise ValueError(
            f"model.num_classes is {configured_count}, but the dataset has "
            f"{len(class_names)} classes/labels."
        )
    model_config["num_classes"] = len(class_names)
    dataset_config = configuration["dataset"]
    resume_dataset_path = (
        dataset_config.get("path")
        or dataset_config.get("train_dir")
        or dataset_config.get("annotations_csv")
    )
    if not resume_dataset_path:
        raise ValueError("Dataset configuration has no path to record for checkpoint resume.")
    resume_dataset_metadata = {
        "format": dataset_config["format"],
        "path": str(Path(resume_dataset_path).expanduser().resolve()),
        "validation_fraction": dataset_config["validation_fraction"],
        "seed": training_config["seed"],
        "image_size": dataset_config["image_size"],
        "resize_enabled": dataset_config["resize_enabled"],
        "mean": list(dataset_config["mean"]),
        "std": list(dataset_config["std"]),
        "augmentation": dataset_config["augmentation"],
    }
    if dataset_config.get("train_dir") and not dataset_config.get("path"):
        resume_dataset_metadata["validation_dir"] = (
            str(Path(dataset_config["validation_dir"]).expanduser().resolve())
            if dataset_config.get("validation_dir")
            else None
        )
    dataset_metadata = _record_dataset_manifest(
        experiment_dir.parent, train_loader, validation_loader, class_names, dataset_metadata
    )
    experiment_record["dataset"] = dataset_metadata
    experiment_record["model"] = {
        "name": model_config["name"],
        "architecture": model_config["name"],
        "pretrained": model_config["pretrained"],
        "num_classes": model_config["num_classes"],
        "adapter": model_adapter.to_dict(),
    }
    experiment_record["environment"].update(
        {"requested_device": training_config["device"]}
    )
    experiment_record["artifacts"]["dataset_manifest"] = dataset_metadata["manifest_path"]
    dump_config(configuration, experiment_dir / "config.yaml")

    def persist_status(
        status: str, epoch: int | None, total_epochs: int | None
    ) -> None:
        experiment_record["status"] = status
        experiment_record["stage"] = status.lower()
        experiment_record["training"]["epoch"] = epoch
        experiment_record["training"]["requested_epochs"] = total_epochs
        if status in {"Stopped", "Completed"}:
            experiment_record["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        _update_run_record(experiment_dir, experiment_record, started_monotonic)
        if task_status_callback is not None:
            task_status_callback(status, epoch, total_epochs)

    try:
        logger.info("Experiment: %s", experiment_dir.name)
        logger.info("Configuration: %s", experiment_dir / "config.yaml")
        logger.info(
            "Task: %s | Model: %s | Requested device: %s",
            task,
            model_config["name"],
            training_config["device"],
        )
        logger.info(
            "Dataset: %s train / %s validation examples",
            dataset_metadata["train_samples"],
            dataset_metadata["validation_samples"],
        )
        model = build_model_from_config(configuration)
        loss_function = get_loss(task, training_config["loss"]["name"])
        optimizer = build_optimizer(
            model.parameters(),
            training_config["optimizer"],
            learning_rate=training_config["learning_rate"],
        )
        metric_functions = (
            {"micro_f1": _multilabel_micro_f1(configuration["evaluation"]["threshold"])}
            if task == "multi_label"
            else {"accuracy": _classification_accuracy}
        )
        early_stopping_config = training_config["early_stopping"]
        if early_stopping_config.get("enabled"):
            early_stopping_config.setdefault("mode", training_config["monitor_mode"])
        trainer = Trainer(
            model,
            loss_function,
            optimizer,
            scheduler=build_scheduler(
                optimizer, training_config["scheduler"], epochs=training_config["epochs"]
            ),
            metrics=metric_functions,
            device=training_config["device"],
            checkpoint_dir=experiment_dir,
            monitor=training_config["monitor"],
            mode=training_config["monitor_mode"],
            early_stopping=build_early_stopping(early_stopping_config),
            amp=training_config["amp"]["enabled"],
            gradient_accumulation_steps=training_config["gradient_accumulation_steps"],
            gradient_clip_norm=training_config["gradient_clip_norm"],
            task_type=task,
            label_names=class_names,
            resume_metadata={
                "dataset": resume_dataset_metadata,
                "model": {
                    "name": model_config["name"],
                    "pretrained": model_config["pretrained"],
                    "num_classes": model_config["num_classes"],
                },
                "training": {
                    "batch_size": training_config["batch_size"],
                    "num_workers": training_config["num_workers"],
                },
            },
            event_callback=event_emitter if event_callback is not None else None,
        )
        if training_config["resume_from"] is not None:
            next_epoch = trainer.resume(training_config["resume_from"])
            if next_epoch > training_config["epochs"]:
                raise ValueError(
                    f"training.epochs ({training_config['epochs']}) must be greater than "
                    f"the resumed checkpoint epoch ({next_epoch - 1}). Increase epochs "
                    "or choose an earlier checkpoint."
                )
            _preserve_best_checkpoint(training_config["resume_from"], experiment_dir)
        _sync_optimizer_config(configuration, trainer)
        dump_config(configuration, experiment_dir / "config.yaml")
        logger.debug("Effective configuration:\n%s", yaml.safe_dump(configuration, sort_keys=False))
        device_name = (
            torch.cuda.get_device_name(trainer.device)
            if trainer.device.type == "cuda"
            else "CPU"
        )
        logger.info("Resolved device: %s (%s)", trainer.device, device_name)
        if training_config["device"] == "auto" and trainer.device.type == "cpu":
            logger.warning("CUDA is unavailable; this experiment is running on CPU.")

        experiment_record["environment"].update(
            {"device": str(trainer.device), "device_name": device_name}
        )
        experiment_record["training"].update(
            {"configuration": training_config, "requested_epochs": training_config["epochs"]}
        )
        _update_run_record(experiment_dir, experiment_record, started_monotonic)
        persist_status("Running", None, training_config["epochs"])
        history = trainer.fit(
            train_loader,
            validation_loader,
            epochs=training_config["epochs"],
            stop_requested=stop_requested,
            status_callback=persist_status,
        )
        if trainer.stopped:
            history = trainer.training_history
            last_epoch = int(history[-1]["epoch"]) if history else trainer._epoch or None
            monitored_rows = [
                row
                for row in history
                if isinstance(row.get(training_config["monitor"]), (int, float))
            ]
            best_row = None
            if monitored_rows:
                best_row = (
                    min(monitored_rows, key=lambda row: row[training_config["monitor"]])
                    if training_config["monitor_mode"] == "min"
                    else max(monitored_rows, key=lambda row: row[training_config["monitor"]])
                )
            persist_status("Stopped", last_epoch, training_config["epochs"])
            experiment_record["training"].update(
                {"epochs_completed": last_epoch or 0, "history": history}
            )
            experiment_record["checkpoints"] = {
                kind: {
                    "path": filename,
                    "epoch": (
                        int(best_row["epoch"])
                        if kind == "best" and best_row
                        else last_epoch
                    ),
                    "monitor": training_config["monitor"] if kind == "best" else None,
                    "value": (
                        best_row.get(training_config["monitor"])
                        if kind == "best" and best_row
                        else None
                    ),
                    "supports_resume": True,
                    "inference_only": False,
                }
                for kind, filename in (("best", "best_model.pth"), ("latest", "last_model.pth"))
                if (experiment_dir / filename).is_file()
            }
            experiment_record["artifacts"]["checkpoints"] = experiment_record["checkpoints"]
            if (experiment_dir / "best_model.pth").is_file():
                asset = _register_experiment_model_asset(
                    experiment_dir, experiment_record, configuration, None
                )
                experiment_record["model_asset_id"] = asset["id"]
                experiment_record["artifacts"]["model_asset"] = (
                    f".uatf/models/{asset['id']}.json"
                )
            _update_run_record(experiment_dir, experiment_record, started_monotonic)
            logger.info("Training stopped at an epoch boundary: %s", experiment_dir)
            return experiment_dir
        history = trainer.training_history
        result = evaluate_training_run(
            trainer.model,
            validation_loader,
            experiment_dir,
            task=task,
            class_names=class_names,
            threshold=configuration["evaluation"]["threshold"],
            device=trainer.device,
            loss_function=loss_function,
            history=history,
            save_predictions=configuration["output"]["save_predictions"],
            save_plots=configuration["output"]["save_plots"],
        )
        completed_at = datetime.now(timezone.utc).isoformat()
        primary_metric = configuration["evaluation"]["primary_metric"]
        primary_history_key = f"val_{primary_metric}"
        primary_history = [
            row for row in history if isinstance(row.get(primary_history_key), (int, float))
        ]
        primary_best = None
        if primary_history:
            maximize = primary_metric != "loss"
            primary_best = (
                max(primary_history, key=lambda row: row[primary_history_key])
                if maximize
                else min(primary_history, key=lambda row: row[primary_history_key])
            )
        monitored_history = [
            row for row in history if isinstance(row.get(training_config["monitor"]), (int, float))
        ]
        best_history = None
        if monitored_history:
            best_history = (
                min(monitored_history, key=lambda row: row[training_config["monitor"]])
                if training_config["monitor_mode"] == "min"
                else max(monitored_history, key=lambda row: row[training_config["monitor"]])
            )
        last_history = history[-1] if history else {}
        checkpoint_records = {
            "best": {
                "path": "best_model.pth",
                "epoch": int(best_history["epoch"]) if best_history else None,
                "monitor": training_config["monitor"],
                "value": best_history.get(training_config["monitor"]) if best_history else None,
                "supports_resume": True,
                "inference_only": False,
            },
            "latest": {
                "path": "last_model.pth",
                "epoch": int(last_history["epoch"]) if last_history else None,
                "supports_resume": True,
                "inference_only": False,
            },
        }
        report = {
            "experiment": {
                "id": experiment_record["id"],
                "name": experiment_dir.name,
                "started_at_utc": started_at,
                "completed_at_utc": completed_at,
                "duration_seconds": round(time.monotonic() - started_monotonic, 3),
                "source_config": str(source_config) if source_config is not None else None,
                "configuration_overrides": dict(configuration_overrides or {}),
            },
            "environment": {
                "uatf": __version__,
                "python": platform.python_version(),
                "pytorch": str(torch.__version__),
                "device": str(trainer.device),
                "device_name": device_name,
            },
            "task": task,
            "model": {
                "name": model_config["name"],
                "pretrained": model_config["pretrained"],
                "num_classes": model_config["num_classes"],
                "adapter": model_adapter.to_dict(),
            },
            "dataset": dataset_metadata,
            "training": {
                "epochs_requested": training_config["epochs"],
                "epochs_completed": trainer.start_epoch - 1,
                "history": history,
            },
            "evaluation": {
                "primary_metric": primary_metric,
                "primary_value": result.metrics.get(primary_metric),
                "best_value": (
                    float(primary_best[primary_history_key]) if primary_best else None
                ),
                "best_epoch": int(primary_best["epoch"]) if primary_best else None,
                "threshold": configuration["evaluation"]["threshold"],
                "loss": result.loss,
                "sample_count": result.metrics["sample_count"],
                "class_names": class_names,
                "metrics": result.metrics,
            },
            "checkpoint": {
                "best": "best_model.pth",
                "last": "last_model.pth",
            },
        }
        _write_json(experiment_dir / "metrics.json", report)
        experiment_record["training"]["epochs_completed"] = trainer.start_epoch - 1
        asset = _register_experiment_model_asset(
            experiment_dir, experiment_record, configuration, report["evaluation"]
        )
        experiment_record["status"] = "Completed"
        experiment_record["finished_at_utc"] = completed_at
        experiment_record["model_asset_id"] = asset["id"]
        experiment_record["dataset"] = dataset_metadata
        experiment_record["model"] = report["model"] | {"architecture": model_config["name"]}
        experiment_record["environment"].update(report["environment"])
        experiment_record["training"].update(
            {
                "epochs_completed": trainer.start_epoch - 1,
                "history": history,
                "configuration": training_config,
            }
        )
        experiment_record["evaluation"] = report["evaluation"]
        experiment_record["duration_seconds"] = report["experiment"]["duration_seconds"]
        experiment_record["checkpoints"] = checkpoint_records
        experiment_record["artifacts"].update(
            {
                "metrics": "metrics.json",
                "plots": "plots" if (experiment_dir / "plots").is_dir() else None,
                "predictions": "predictions.csv"
                if (experiment_dir / "predictions.csv").is_file()
                else None,
                "checkpoints": checkpoint_records,
                    "model_asset": f".uatf/models/{asset['id']}.json",
            }
        )
        _update_run_record(experiment_dir, experiment_record, started_monotonic)
        if event_callback is not None:
            event_emitter(
                TrainingEvent(
                    event_type="training_completed",
                    experiment_dir=str(experiment_dir),
                    epoch=int(history[-1]["epoch"]) if history else None,
                    total_epochs=training_config["epochs"],
                    metrics={
                        key: float(value)
                        for key, value in result.metrics.items()
                        if isinstance(value, (int, float))
                    },
                )
            )
        logger.info("Best checkpoint: %s", experiment_dir / "best_model.pth")
        logger.info("Metrics: %s", experiment_dir / "metrics.json")
        logger.info("Experiment log: %s", log_path)
        logger.info("Training completed: %s", experiment_dir)
    except Exception:
        logger.exception("Training failed for experiment %s", experiment_dir.name)
        raise
    finally:
        shutdown_logging()
    return experiment_dir
