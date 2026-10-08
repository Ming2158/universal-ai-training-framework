"""Configuration-driven image classification training command."""

from __future__ import annotations

import argparse
import json
import platform
import random
import sys
import traceback
from collections.abc import Mapping
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TextIO

import torch
import yaml
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from uatf.config import load_config, merge_config, validate_config
from uatf.data import (
    CsvImageClassificationDataset,
    CsvMultiLabelDataset,
    ImageFolderDataset,
    build_image_transform,
    create_data_loaders,
)
from uatf.evaluation import evaluate_training_run
from uatf.models import build_model_from_config, get_loss
from uatf.training import Trainer, build_early_stopping, build_scheduler


class _Tee:
    """Send command output to both the terminal and the experiment log."""

    def __init__(self, terminal: TextIO, log_file: TextIO) -> None:
        self.terminal = terminal
        self.log_file = log_file

    def write(self, value: str) -> int:
        self.terminal.write(value)
        self.log_file.write(value)
        return len(value)

    def flush(self) -> None:
        self.terminal.flush()
        self.log_file.flush()


def _path(value: str | None) -> Path | None:
    return Path(value).expanduser().resolve() if value is not None else None


def _image_loaders(
    configuration: Mapping[str, Any], task: str, seed: int
) -> tuple[DataLoader, DataLoader, list[str], dict[str, Any]]:
    dataset_config = configuration["dataset"]
    training_config = configuration["training"]
    transform = build_image_transform(
        image_size=dataset_config["image_size"],
        mean=dataset_config["mean"],
        std=dataset_config["std"],
    )
    batch_size = training_config["batch_size"]
    num_workers = training_config["num_workers"]
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

        if train_path is None:
            raise ValueError("Image-folder data requires dataset.path or dataset.train_dir.")
        train_dataset = ImageFolderDataset(train_path, transform=transform)
        if validation_path is not None:
            validation_dataset = ImageFolderDataset(validation_path, transform=transform)
            if train_dataset.classes != validation_dataset.classes:
                raise ValueError(
                    "Training and validation folders must contain the same class directories "
                    "in the same sorted order."
                )
            train_loader = _loader(
                train_dataset, batch_size, num_workers, shuffle=True, seed=seed
            )
            validation_loader = _loader(
                validation_dataset, batch_size, num_workers, shuffle=False, seed=seed
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
        )
    else:
        if labels is not None:
            raise ValueError("dataset.label_columns is only supported for csv_multilabel.")
        full_dataset = CsvImageClassificationDataset(
            annotation_path, image_root=image_root, transform=transform
        )

    train_loader, validation_loader, _ = create_data_loaders(
        full_dataset,
        batch_size=batch_size,
        validation_fraction=dataset_config["validation_fraction"],
        seed=seed,
        num_workers=num_workers,
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
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        generator=torch.Generator().manual_seed(seed) if shuffle else None,
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
    with path.open("w", encoding="utf-8") as destination:
        json.dump(value, destination, indent=2, ensure_ascii=False, allow_nan=False)
        destination.write("\n")


def _run(configuration: dict[str, Any], source_config: Path) -> Path:
    task = configuration["task"]["type"]
    training_config = configuration["training"]
    seed = training_config["seed"]
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
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

    experiment_dir = _experiment_directory(configuration["output"])
    (experiment_dir / "logs").mkdir()
    with (experiment_dir / "config.yaml").open("w", encoding="utf-8") as destination:
        yaml.safe_dump(configuration, destination, sort_keys=False, allow_unicode=True)

    log_path = experiment_dir / "logs" / "train.log"
    original_stdout, original_stderr = sys.stdout, sys.stderr
    try:
        with log_path.open("w", encoding="utf-8") as log_file:
            stdout_tee, stderr_tee = _Tee(original_stdout, log_file), _Tee(original_stderr, log_file)
            with redirect_stdout(stdout_tee), redirect_stderr(stderr_tee):
                started_at = datetime.now(timezone.utc).isoformat()
                print(f"Experiment: {experiment_dir.name}")
                print(f"Configuration: {experiment_dir / 'config.yaml'}")
                print(f"Task: {task} | Model: {model_config['name']} | Device: {training_config['device']}")
                print(
                    f"Dataset: {dataset_metadata['train_samples']} train / "
                    f"{dataset_metadata['validation_samples']} validation examples"
                )

                model = build_model_from_config(configuration)
                loss_function = get_loss(task)
                optimizer = torch.optim.Adam(
                    model.parameters(), lr=training_config["learning_rate"]
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
                    scheduler=build_scheduler(optimizer, training_config["scheduler"]),
                    metrics=metric_functions,
                    device=training_config["device"],
                    checkpoint_dir=experiment_dir,
                    monitor=training_config["monitor"],
                    mode=training_config["monitor_mode"],
                    early_stopping=build_early_stopping(early_stopping_config),
                    amp=training_config["amp"]["enabled"],
                )
                history = trainer.fit(
                    train_loader,
                    validation_loader,
                    epochs=training_config["epochs"],
                )
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
                )
                completed_at = datetime.now(timezone.utc).isoformat()
                primary_metric = configuration["evaluation"]["primary_metric"]
                device_name = (
                    torch.cuda.get_device_name(trainer.device)
                    if trainer.device.type == "cuda"
                    else "CPU"
                )
                report = {
                    "experiment": {
                        "name": experiment_dir.name,
                        "started_at_utc": started_at,
                        "completed_at_utc": completed_at,
                        "source_config": str(source_config),
                    },
                    "environment": {
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
                    },
                    "dataset": dataset_metadata,
                    "training": {
                        "epochs_requested": training_config["epochs"],
                        "epochs_completed": len(history),
                        "history": history,
                    },
                    "evaluation": {
                        "primary_metric": primary_metric,
                        "primary_value": result.metrics.get(primary_metric),
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
                print(f"Best checkpoint: {experiment_dir / 'best_model.pth'}")
                print(f"Metrics: {experiment_dir / 'metrics.json'}")
                print(f"Experiment log: {log_path}")
    except Exception:
        with log_path.open("a", encoding="utf-8") as log_file:
            log_file.write("\nTraining failed:\n")
            log_file.write(traceback.format_exc())
        raise
    return experiment_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="用 YAML 配置运行 UATF 图像分类训练。")
    parser.add_argument("--config", required=True, type=Path, help="YAML 配置文件路径")
    parser.add_argument("--epochs", type=int, help="覆盖配置中的训练轮数")
    parser.add_argument("--batch-size", type=int, help="覆盖配置中的批次大小")
    parser.add_argument("--learning-rate", type=float, help="覆盖配置中的学习率")
    parser.add_argument("--seed", type=int, help="覆盖配置中的随机种子")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), help="覆盖运行设备")
    parser.add_argument("--output-dir", type=str, help="覆盖实验输出根目录")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        user_config = load_config(args.config)
        configuration = dict(user_config)
        cli_training: dict[str, Any] = {}
        for argument, key in (
            ("epochs", "epochs"),
            ("batch_size", "batch_size"),
            ("learning_rate", "learning_rate"),
            ("seed", "seed"),
            ("device", "device"),
        ):
            value = getattr(args, argument)
            if value is not None:
                cli_training[key] = value
        if cli_training:
            user_training = user_config.get("training", {})
            configuration["training"] = (
                merge_config(user_training, cli_training)
                if isinstance(user_training, Mapping)
                else user_training
            )
        if args.output_dir is not None:
            user_output = user_config.get("output", {})
            configuration["output"] = (
                merge_config(user_output, {"directory": args.output_dir})
                if isinstance(user_output, Mapping)
                else user_output
            )
        validated = validate_config(configuration)
        experiment_dir = _run(validated, args.config.expanduser().resolve())
        print(f"Training completed: {experiment_dir}")
        return 0
    except (FileNotFoundError, FileExistsError, ValueError, OSError, RuntimeError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
