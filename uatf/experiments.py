"""Read and manage local experiment records and model assets."""

from __future__ import annotations

import json
import os
import uuid
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import yaml


def _root(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def _inside(root: Path, value: str | Path) -> Path:
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = root / candidate
    resolved = candidate.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise ValueError(f"Path is outside the UATF output directory: {resolved}") from error
    return resolved


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        resolved = path.resolve()
        resolved.relative_to(path.parent.resolve())
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, RuntimeError, ValueError, UnicodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _read_yaml(path: Path) -> dict[str, Any] | None:
    try:
        resolved = path.resolve()
        resolved.relative_to(path.parent.resolve())
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, RuntimeError, ValueError, UnicodeError, yaml.YAMLError):
        return None
    return value if isinstance(value, dict) else None


def _rooted_json(root: Path, path: Path) -> dict[str, Any] | None:
    try:
        return _read_json(_inside(root, path))
    except ValueError:
        return None


def _rooted_yaml(root: Path, path: Path) -> dict[str, Any] | None:
    try:
        return _read_yaml(_inside(root, path))
    except ValueError:
        return None


def _experiment_directories(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    directories: list[Path] = []
    for path in root.iterdir():
        if path.name == ".uatf" or not path.is_dir():
            continue
        try:
            resolved = _inside(root, path)
        except ValueError:
            continue
        if any((resolved / filename).is_file() for filename in ("experiment.json", "config.yaml", "metrics.json")):
            directories.append(resolved)
    return sorted(directories, key=lambda path: path.name.casefold())


def _record_for(directory: Path) -> dict[str, Any]:
    root = directory.parent
    record = _rooted_json(root, directory / "experiment.json")
    if record is not None:
        record.setdefault("id", directory.name)
        record.setdefault("name", directory.name)
        record.setdefault("status", "Unknown")
        return record

    config = _rooted_yaml(root, directory / "config.yaml") or {}
    metrics = _rooted_json(root, directory / "metrics.json") or {}
    experiment = metrics.get("experiment", {})
    evaluation = metrics.get("evaluation", {})
    status = "Completed" if metrics else "Unknown"
    failure = None
    tasks_directory = root / ".uatf" / "tasks"
    try:
        tasks_directory = _inside(root, tasks_directory)
    except ValueError:
        tasks_directory = root / ".uatf" / "missing-tasks"
    if not metrics and tasks_directory.is_dir():
        for task_path in tasks_directory.glob("*.json"):
            task = _rooted_json(root, task_path)
            if not task or not task.get("result_directory"):
                continue
            try:
                same_directory = Path(task["result_directory"]).resolve() == directory.resolve()
            except (OSError, RuntimeError, ValueError):
                same_directory = False
            if same_directory:
                status = str(task.get("status") or status)
                failure = task.get("error_summary")
                break
    return {
        "schema_version": 0,
        "id": str(experiment.get("id") or directory.name),
        "name": str(experiment.get("name") or directory.name),
        "created_at_utc": experiment.get("started_at_utc"),
        "status": status,
        "legacy": True,
        "directory": directory.name,
        "task": metrics.get("task") or config.get("task", {}).get("type"),
        "dataset": metrics.get("dataset", {}),
        "model": metrics.get("model", config.get("model", {})),
        "evaluation": evaluation,
        "training": metrics.get("training", {}),
        "failure": {"message": failure} if failure else None,
        "artifacts": {},
    }


def list_experiments(
    output_directory: str | Path,
    *,
    name: str | None = None,
    model: str | None = None,
    status: str | None = None,
    created_after: str | None = None,
    created_before: str | None = None,
) -> list[dict[str, Any]]:
    """List local runs, including V1 directories that have no experiment manifest."""
    root = _root(output_directory)
    records: list[dict[str, Any]] = []
    for directory in _experiment_directories(root):
        record = _record_for(directory)
        config = _rooted_yaml(root, directory / "config.yaml") or {}
        experiment = record.get("experiment", {})
        dataset = record.get("dataset", {})
        model_info = record.get("model", {})
        evaluation = record.get("evaluation", {})
        created_at = str(record.get("created_at_utc") or experiment.get("started_at_utc") or "")
        model_name = str(model_info.get("name") or config.get("model", {}).get("name") or "")
        run_name = str(record.get("name") or directory.name)
        run_status = str(record.get("status") or "Unknown")
        if name and name.casefold() not in run_name.casefold():
            continue
        if model and model.casefold() not in model_name.casefold():
            continue
        if status and run_status.casefold() != status.casefold():
            continue
        if created_after and created_at and created_at < created_after:
            continue
        if created_before and created_at and created_at > created_before:
            continue

        summary = {
            "id": str(record.get("id") or directory.name),
            "name": run_name,
            "directory": directory.name,
            "created_at_utc": created_at or None,
            "status": run_status,
            "task": record.get("task") or config.get("task", {}).get("type"),
            "model": model_name or None,
            "dataset_id": dataset.get("id") or dataset.get("fingerprint"),
            "duration_seconds": record.get("duration_seconds"),
            "primary_metric": evaluation.get("primary_metric"),
            "best_metric": evaluation.get("best_value"),
            "legacy": bool(record.get("legacy")),
            "path": str(directory),
        }
        if summary["best_metric"] is None:
            summary["best_metric"] = evaluation.get("primary_value")
        records.append(summary)
    return sorted(records, key=lambda item: str(item["created_at_utc"] or ""), reverse=True)


def get_experiment(output_directory: str | Path, experiment_id: str) -> dict[str, Any]:
    """Load one experiment's record, configs, metrics, logs, and checkpoints."""
    root = _root(output_directory)
    if not experiment_id or Path(experiment_id).name != experiment_id:
        raise ValueError("Experiment id must be a single directory identifier.")
    selected: Path | None = None
    for directory in _experiment_directories(root):
        record = _record_for(directory)
        if experiment_id in {directory.name, str(record.get("id"))}:
            selected = directory
            break
    if selected is None:
        raise FileNotFoundError(f"Experiment does not exist: {experiment_id}")

    record = _record_for(selected)
    config = _rooted_yaml(root, selected / "config.yaml")
    source_config = _rooted_yaml(root, selected / "source_config.yaml")
    metrics = _rooted_json(root, selected / "metrics.json")
    log_path = _inside(root, selected / "logs" / "train.log")
    log_text = ""
    if log_path.is_file():
        try:
            with log_path.open("rb") as source:
                source.seek(0, os.SEEK_END)
                source.seek(max(0, source.tell() - 128 * 1024))
                log_text = source.read().decode("utf-8", errors="replace")
        except OSError:
            pass

    checkpoint_rows: list[dict[str, Any]] = []
    for kind, filename in (("best", "best_model.pth"), ("latest", "last_model.pth")):
        path = _inside(root, selected / filename)
        if not path.is_file():
            continue
        stat = path.stat()
        checkpoint_rows.append(
            {
                "kind": kind,
                "path": str(path),
                "size_bytes": stat.st_size,
                "saved_at_utc": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
                "epoch": record.get("checkpoints", {}).get(kind, {}).get("epoch"),
                "supports_resume": True,
                "inference_only": False,
            }
        )
    return {
        "record": record,
        "configuration": config,
        "source_configuration": source_config,
        "metrics": metrics,
        "history": (metrics or {}).get("training", {}).get(
            "history", record.get("training", {}).get("history", [])
        ),
        "log": log_text,
        "checkpoints": checkpoint_rows,
        "directory": str(selected),
    }


def compare_experiments(
    output_directory: str | Path, experiment_ids: list[str]
) -> dict[str, Any]:
    """Compare selected runs and identify whether their validation scores align."""
    if len(experiment_ids) < 2:
        raise ValueError("Select at least two experiments to compare.")
    details = [get_experiment(output_directory, experiment_id) for experiment_id in experiment_ids]
    records = [item["record"] for item in details]
    configs = [item["configuration"] or {} for item in details]

    def metric_definition(record: dict[str, Any], config: dict[str, Any]) -> tuple[Any, Any]:
        evaluation = record.get("evaluation", {})
        config_evaluation = config.get("evaluation", {})
        return (
            evaluation.get("primary_metric") or config_evaluation.get("primary_metric"),
            evaluation.get("threshold", config_evaluation.get("threshold")),
        )

    task_values = [record.get("task") or config.get("task", {}).get("type") for record, config in zip(records, configs)]
    dataset_values = [
        (record.get("dataset", {}).get("id"), record.get("dataset", {}).get("split_id"))
        for record in records
    ]
    metric_values = [metric_definition(record, config) for record, config in zip(records, configs)]
    warnings: list[str] = []
    if len(set(task_values)) != 1:
        warnings.append("任务类型不同，验证指标不能直接比较。")
    if len(set(dataset_values)) != 1 or any(not value[0] or not value[1] for value in dataset_values):
        warnings.append("数据集或验证划分不同/信息不完整，验证指标不能直接比较。")
    if len(set(metric_values)) != 1 or any(not value[0] for value in metric_values):
        warnings.append("主要指标定义或阈值不同/信息不完整，验证指标不能直接比较。")
    if any(
        record.get("evaluation", {}).get("best_value") is None
        and record.get("evaluation", {}).get("primary_value") is None
        for record in records
    ):
        warnings.append("至少一个实验缺少验证指标结果，不能直接比较分数。")

    rows: list[dict[str, Any]] = []
    for detail, record, config in zip(details, records, configs):
        evaluation = record.get("evaluation", {})
        model = record.get("model", {}) or config.get("model", {})
        dataset = record.get("dataset", {})
        rows.append(
            {
                "id": record.get("id"),
                "name": record.get("name"),
                "dataset_id": dataset.get("id"),
                "split_id": dataset.get("split_id"),
                "task": record.get("task") or config.get("task", {}).get("type"),
                "model": model.get("name"),
                "pretrained": model.get("pretrained"),
                "training": config.get("training", {}),
                "metric": evaluation.get("primary_metric"),
                "best_validation_metric": evaluation.get("best_value", evaluation.get("primary_value")),
                "duration_seconds": record.get("duration_seconds"),
                "checkpoints": detail["checkpoints"],
                "result_directory": detail["directory"],
            }
        )
    return {"comparable": not warnings, "warnings": warnings, "experiments": rows}


def register_model_asset(output_directory: str | Path, asset: dict[str, Any]) -> dict[str, Any]:
    """Persist a metadata-only model asset reference to an experiment checkpoint."""
    root = _root(output_directory)
    asset_id = str(asset.get("id") or f"model-{uuid.uuid4().hex}")
    if Path(asset_id).name != asset_id:
        raise ValueError("Model asset id must be a single identifier.")
    weights = _inside(root, str(asset["weights_path"]))
    config = _inside(root, str(asset["configuration_path"]))
    if not weights.is_file() or not config.is_file():
        raise FileNotFoundError("Model asset weights or configuration do not exist.")
    relative_weights = weights.relative_to(root).as_posix()
    relative_config = config.relative_to(root).as_posix()
    registry = root / ".uatf" / "models"
    registry.mkdir(parents=True, exist_ok=True)
    record = {**asset, "id": asset_id, "weights_path": relative_weights, "configuration_path": relative_config}
    destination = registry / f"{asset_id}.json"
    temporary = destination.with_name(f"{destination.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(record, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return record


def list_model_assets(output_directory: str | Path) -> list[dict[str, Any]]:
    root = _root(output_directory)
    registry = root / ".uatf" / "models"
    try:
        registry = _inside(root, registry)
    except ValueError:
        registry = root / ".uatf" / "missing-models"
    assets: list[dict[str, Any]] = []
    registered_experiments: set[str] = set()
    if registry.is_dir():
        for path in registry.glob("*.json"):
            asset = _rooted_json(root, path)
            if asset is None:
                continue
            try:
                weights = _inside(root, str(asset.get("weights_path", "")))
            except ValueError:
                asset["available"] = False
            else:
                try:
                    configuration = _inside(root, str(asset.get("configuration_path", "")))
                except ValueError:
                    asset["available"] = False
                else:
                    asset["available"] = weights.is_file() and configuration.is_file()
                    saved_configuration = _rooted_yaml(root, configuration) or {}
                    saved_dataset = saved_configuration.get("dataset", {})
                    asset.setdefault("dataset_format", saved_dataset.get("format"))
                    asset.setdefault(
                        "preprocessing",
                        {
                            key: saved_dataset.get(key)
                            for key in ("image_size", "mean", "std")
                        },
                    )
                asset["weights_path"] = str(weights)
            if asset.get("experiment_id"):
                registered_experiments.add(str(asset["experiment_id"]))
            assets.append(asset)

    for directory in _experiment_directories(root):
        record = _record_for(directory)
        experiment_id = str(record.get("id") or directory.name)
        weights = directory / "best_model.pth"
        config_path = directory / "config.yaml"
        if (
            experiment_id in registered_experiments
            or not weights.is_file()
            or not config_path.is_file()
        ):
            continue
        config = _rooted_yaml(root, config_path) or {}
        metrics = _rooted_json(root, directory / "metrics.json") or {}
        model = record.get("model") or config.get("model", {})
        assets.append(
            {
                "id": f"legacy-model-{experiment_id}",
                "name": record.get("name") or directory.name,
                "architecture": model.get("name"),
                "pretrained": model.get("pretrained"),
                "experiment_id": experiment_id,
                "experiment_directory": directory.name,
                "weights_path": str(weights.resolve()),
                "configuration_path": str(config_path.resolve()),
                "task": record.get("task") or config.get("task", {}).get("type"),
                "dataset_format": config.get("dataset", {}).get("format"),
                "preprocessing": {
                    key: config.get("dataset", {}).get(key) for key in ("image_size", "mean", "std")
                },
                "training_summary": record.get("training", {}),
                "evaluation": record.get("evaluation") or metrics.get("evaluation", {}),
                "created_at_utc": record.get("created_at_utc"),
                "version": 0,
                "weights_kind": "legacy_full_training_checkpoint",
                "legacy": True,
                "available": True,
            }
        )
    for asset in assets:
        if not asset.get("class_names"):
            asset["class_names"] = _asset_class_names(root, asset)
    return sorted(assets, key=lambda asset: str(asset.get("created_at_utc") or ""), reverse=True)


def _asset_class_names(root: Path, asset: dict[str, Any]) -> list[str] | None:
    names = asset.get("class_names")
    if isinstance(names, list) and names and all(isinstance(name, str) and name for name in names):
        return names

    experiment_directory = asset.get("experiment_directory")
    if experiment_directory:
        try:
            directory = _inside(root, str(experiment_directory))
        except ValueError:
            directory = None
        if directory is not None:
            record = _read_json(directory / "experiment.json") or {}
            dataset = record.get("dataset", {})
            names = dataset.get("class_names")
            if isinstance(names, list) and names and all(isinstance(name, str) and name for name in names):
                return names
            manifest_path = dataset.get("manifest_path")
            if manifest_path:
                try:
                    manifest = _read_json(_inside(root, str(manifest_path))) or {}
                except ValueError:
                    manifest = {}
                names = manifest.get("class_names")
                if isinstance(names, list) and names and all(
                    isinstance(name, str) and name for name in names
                ):
                    return names
            names = record.get("evaluation", {}).get("class_names")
            if isinstance(names, list) and names and all(isinstance(name, str) and name for name in names):
                return names
            metrics = _read_json(directory / "metrics.json") or {}
            evaluation = metrics.get("evaluation", metrics)
            names = evaluation.get("class_names") if isinstance(evaluation, dict) else None
            if isinstance(names, list) and names and all(isinstance(name, str) and name for name in names):
                return names
    labels = asset.get("configuration", {}).get("dataset", {}).get("label_columns")
    if isinstance(labels, list) and labels and all(isinstance(name, str) and name for name in labels):
        return labels
    return None


def load_model_asset_bundle(output_directory: str | Path, asset_id: str) -> dict[str, Any]:
    """Load an asset's model, configuration, and label metadata through one path."""
    root = _root(output_directory)
    if not asset_id or Path(asset_id).name != asset_id:
        raise ValueError("Model asset id must be a single identifier.")
    asset = next((item for item in list_model_assets(root) if item.get("id") == asset_id), None)
    if asset is None:
        raise FileNotFoundError(f"Model asset is not registered: {asset_id}")
    if not asset.get("available"):
        raise FileNotFoundError(f"Model asset weights are unavailable: {asset_id}")
    weights_path = _inside(root, str(asset.get("weights_path", "")))
    config_path = _inside(root, str(asset.get("configuration_path", "")))
    if not weights_path.is_file() or not config_path.is_file():
        raise FileNotFoundError("Model asset weights or configuration are missing.")
    configuration = _read_yaml(config_path)
    if configuration is None:
        raise ValueError(f"Could not read model asset configuration: {config_path}")
    from uatf.models import build_model_from_config
    from uatf.models.factory import normalize_task

    model_config = configuration.get("model", {})
    task_config = configuration.get("task", {})
    if not isinstance(model_config, dict):
        raise ValueError("Model asset configuration has an invalid model section.")
    configured_task = (
        task_config
        if isinstance(task_config, str)
        else task_config.get("type")
        if isinstance(task_config, dict)
        else None
    )
    if asset.get("architecture") and str(asset["architecture"]).lower() != str(
        model_config.get("name", "")
    ).lower():
        raise ValueError("Model asset architecture metadata does not match its configuration.")
    if asset.get("task") and normalize_task(str(asset["task"])) != normalize_task(
        str(configured_task or "classification")
    ):
        raise ValueError("Model asset task metadata does not match its configuration.")

    model_configuration = deepcopy(configuration)
    if isinstance(model_configuration.get("model"), dict):
        model_configuration["model"]["pretrained"] = False
    model = build_model_from_config(model_configuration)
    checkpoint = torch.load(weights_path, map_location="cpu", weights_only=True)
    state = checkpoint.get("model_state_dict") if isinstance(checkpoint, dict) else None
    if not isinstance(state, dict):
        raise ValueError(f"Registered weights do not contain a model state dict: {weights_path}")
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as error:
        raise ValueError(f"Model asset structure does not match its weights: {error}") from error
    model.eval()
    asset["class_names"] = _asset_class_names(root, asset)
    asset["weights_path"] = str(weights_path)
    asset["configuration_path"] = str(config_path)
    return {"model": model, "configuration": configuration, "asset": asset}


def load_model_asset(output_directory: str | Path, asset_id: str):
    """Build an architecture from the saved configuration and load its registered weights."""
    return load_model_asset_bundle(output_directory, asset_id)["model"]


def load_exported_model_asset(weights_path: str | Path) -> dict[str, Any]:
    """Reload an inference state dict with the configuration in its JSON sidecar."""
    path = Path(weights_path).expanduser().resolve()
    sidecar_path = path.with_suffix(path.suffix + ".json")
    if not path.is_file() or not sidecar_path.is_file():
        raise FileNotFoundError("Exported model weights or their JSON sidecar are missing.")
    sidecar = _read_json(sidecar_path)
    if sidecar is None or not isinstance(sidecar.get("configuration"), dict):
        raise ValueError(f"Export sidecar is missing a valid configuration: {sidecar_path}")
    from uatf.models import build_model_from_config
    from uatf.models.factory import normalize_task

    configuration = sidecar["configuration"]
    model_configuration = deepcopy(configuration)
    model_config = configuration.get("model", {})
    dataset_config = configuration.get("dataset", {})
    if not isinstance(model_config, dict) or not isinstance(dataset_config, dict):
        raise ValueError("Exported model configuration has invalid model or dataset settings.")
    if sidecar.get("architecture") and str(sidecar["architecture"]).lower() != str(
        model_config.get("name", "")
    ).lower():
        raise ValueError("Export sidecar architecture does not match its configuration.")
    if sidecar.get("preprocessing") != dataset_config:
        raise ValueError("Export sidecar preprocessing does not match its model configuration.")
    task_config = configuration.get("task", {})
    configured_task = (
        task_config
        if isinstance(task_config, str)
        else task_config.get("type")
        if isinstance(task_config, dict)
        else "classification"
    )
    if sidecar.get("task") and normalize_task(str(sidecar["task"])) != normalize_task(
        str(configured_task)
    ):
        raise ValueError("Export sidecar task does not match its model configuration.")
    exported_names = sidecar.get("class_names")
    output_count = model_config.get("num_classes", model_config.get("num_labels"))
    if exported_names is not None and (
        not isinstance(exported_names, list)
        or not exported_names
        or len(exported_names) != output_count
        or any(not isinstance(name, str) or not name for name in exported_names)
        or len(set(exported_names)) != len(exported_names)
    ):
        raise ValueError("Export sidecar label names do not match the model output count.")
    if isinstance(model_configuration.get("model"), dict):
        model_configuration["model"]["pretrained"] = False
    model = build_model_from_config(model_configuration)
    state = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict):
        raise ValueError(f"Exported weights are not a state dict: {path}")
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as error:
        raise ValueError(f"Exported model structure does not match its weights: {error}") from error
    model.eval()
    return {"model": model, "configuration": configuration, "asset": sidecar}


def export_model_asset(
    output_directory: str | Path, asset_id: str, destination: str | Path | None = None
) -> Path:
    """Export a registered model as inference-only state-dict weights and a sidecar."""
    root = _root(output_directory)
    assets = {asset["id"]: asset for asset in list_model_assets(root)}
    asset = assets.get(asset_id)
    if asset is None or not asset.get("available"):
        raise FileNotFoundError(f"Model asset is unavailable: {asset_id}")
    source = _inside(root, asset["weights_path"])
    configuration_path = _inside(root, asset["configuration_path"])
    configuration = _read_yaml(configuration_path)
    if configuration is None:
        raise ValueError(f"Could not read model asset configuration: {configuration_path}")
    checkpoint = torch.load(source, map_location="cpu", weights_only=True)
    state = checkpoint.get("model_state_dict") if isinstance(checkpoint, dict) else None
    if not isinstance(state, dict):
        raise ValueError(f"Registered weights do not contain a model state dict: {source}")
    dataset_config = configuration.get("dataset", {})
    if not isinstance(dataset_config, dict):
        raise ValueError("Model asset configuration has an invalid dataset section.")
    task_configuration = configuration.get("task", {"type": asset.get("task") or "classification"})
    if isinstance(task_configuration, str):
        task_name = task_configuration
    elif isinstance(task_configuration, dict):
        task_name = task_configuration.get("type", asset.get("task") or "classification")
    else:
        task_name = asset.get("task") or "classification"
    model_configuration = configuration.get("model", {})
    if not isinstance(model_configuration, dict):
        raise ValueError("Model asset configuration has an invalid model section.")
    export_configuration = {
        "task": {"type": task_name},
        "model": dict(model_configuration),
        "dataset": {
            "format": dataset_config.get(
                "format", "csv_multilabel" if task_name == "multi_label" else "image_folder"
            ),
            "image_size": dataset_config.get("image_size", 224),
            "mean": dataset_config.get("mean", [0.485, 0.456, 0.406]),
            "std": dataset_config.get("std", [0.229, 0.224, 0.225]),
        },
    }
    if "label_columns" in dataset_config:
        export_configuration["dataset"]["label_columns"] = dataset_config["label_columns"]
    if not export_configuration["model"].get("num_classes"):
        export_configuration["model"]["num_classes"] = len(asset.get("class_names") or []) or None
    if not export_configuration["model"].get("num_classes"):
        raise ValueError("Model asset does not contain its output label count.")
    if destination is None:
        export_root = root / ".uatf" / "exports"
        export_root.mkdir(parents=True, exist_ok=True)
        target = export_root / f"{asset_id}.weights.pth"
        suffix = 2
        while target.exists() or target.with_suffix(target.suffix + ".json").exists():
            target = export_root / f"{asset_id}.weights-{suffix}.pth"
            suffix += 1
    else:
        target = Path(destination).expanduser().resolve()
        if target.exists() or target.with_suffix(target.suffix + ".json").exists():
            raise FileExistsError(f"Export destination already exists: {target}")
        target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f"{target.name}.{uuid.uuid4().hex}.tmp")
    try:
        torch.save(state, temporary)
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    sidecar = target.with_suffix(target.suffix + ".json")
    sidecar_temp = sidecar.with_name(f"{sidecar.name}.{uuid.uuid4().hex}.tmp")
    try:
        sidecar_temp.write_text(
            json.dumps(
                {
                    "asset_id": asset_id,
                    "task": asset.get("task"),
                    "architecture": asset.get("architecture"),
                    "class_names": asset.get("class_names"),
                    "configuration": export_configuration,
                    "preprocessing": export_configuration["dataset"],
                    "source_experiment_id": asset.get("experiment_id"),
                    "weights_format": "inference_state_dict",
                    "version": 2,
                    "created_at_utc": datetime.now(timezone.utc).isoformat(),
                },
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        sidecar_temp.replace(sidecar)
    finally:
        sidecar_temp.unlink(missing_ok=True)
    return target


__all__ = [
    "compare_experiments",
    "export_model_asset",
    "get_experiment",
    "list_experiments",
    "list_model_assets",
    "load_model_asset",
    "load_model_asset_bundle",
    "load_exported_model_asset",
    "register_model_asset",
]
