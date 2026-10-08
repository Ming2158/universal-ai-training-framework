"""YAML loading, defaults, and early validation for training configuration."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from math import isfinite
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG: dict[str, Any] = {
    "task": {"type": "classification"},
    "dataset": {
        "validation_fraction": 0.2,
        "image_size": 224,
    },
    "model": {"name": "resnet50", "pretrained": False},
    "training": {
        "seed": 42,
        "epochs": 10,
        "batch_size": 32,
        "learning_rate": 0.001,
        "num_workers": 0,
        "device": "auto",
        "monitor": "val_loss",
        "amp": {"enabled": False},
        "scheduler": {
            "enabled": False,
            "name": "step_lr",
            "step_size": 5,
            "gamma": 0.1,
        },
        "early_stopping": {"enabled": False, "patience": 5, "min_delta": 0.0},
    },
    "evaluation": {"primary_metric": "accuracy", "threshold": 0.5},
    "output": {"directory": "outputs"},
}


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML configuration whose top-level value is a mapping."""
    config_path = Path(path).expanduser()
    try:
        with config_path.open("r", encoding="utf-8") as source:
            configuration = yaml.safe_load(source)
    except FileNotFoundError as error:
        raise FileNotFoundError(f"Configuration file does not exist: {config_path}") from error
    except yaml.YAMLError as error:
        raise ValueError(f"Invalid YAML in configuration {config_path}: {error}") from error

    if not isinstance(configuration, dict):
        raise ValueError(f"Configuration must contain a YAML mapping: {config_path}")
    return configuration


def merge_config(
    defaults: Mapping[str, Any], user_config: Mapping[str, Any]
) -> dict[str, Any]:
    """Recursively merge user values over defaults without mutating either input."""
    result = deepcopy(dict(defaults))
    for key, value in user_config.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = merge_config(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def validate_config(configuration: Mapping[str, Any]) -> dict[str, Any]:
    """Validate known options and return a normalized, complete configuration."""
    from uatf.models.factory import MODEL_REGISTRY, normalize_task

    config = merge_config(DEFAULT_CONFIG, configuration)
    allowed_sections = {"task", "dataset", "model", "training", "evaluation", "output"}
    unknown_sections = sorted(set(config).difference(allowed_sections))
    if unknown_sections:
        raise ValueError(f"Unknown configuration section: {unknown_sections[0]}.")

    for section in allowed_sections:
        if not isinstance(config.get(section), Mapping):
            raise ValueError(f"Configuration section '{section}' must be a mapping.")

    raw_task = configuration.get("task", {})
    task_config = dict(config["task"])
    dataset = dict(config["dataset"])
    dataset_task = dataset.pop("type", None)
    requested_task = raw_task.get("type") if isinstance(raw_task, Mapping) else None
    if dataset_task is not None and requested_task is not None:
        if normalize_task(dataset_task) != normalize_task(requested_task):
            raise ValueError("dataset.type and task.type specify different task types.")
    task_value = dataset_task if dataset_task is not None else requested_task
    if task_value is None:
        task_value = task_config.get("type", "classification")
    task = normalize_task(task_value)
    task_config = {"type": task}

    model = dict(config["model"])
    training = dict(config["training"])
    evaluation = dict(config["evaluation"])
    output = dict(config["output"])

    _check_keys("task", raw_task if isinstance(raw_task, Mapping) else {}, {"type"})
    _check_keys(
        "dataset",
        dataset,
        {
            "format", "path", "train_dir", "validation_dir", "annotations_csv", "image_dir",
            "label_columns", "validation_fraction", "image_size", "mean", "std",
        },
    )
    _check_keys("model", model, {"name", "pretrained", "num_classes", "num_labels"})
    _check_keys(
        "training",
        training,
        {
            "seed", "epochs", "batch_size", "learning_rate", "num_workers", "device",
            "monitor", "monitor_mode", "amp", "scheduler", "early_stopping",
        },
    )
    _check_keys("evaluation", evaluation, {"primary_metric", "threshold"})
    _check_keys("output", output, {"directory", "experiment_name"})

    if not isinstance(model.get("name"), str) or not model["name"].strip():
        raise ValueError("Configuration section 'model' requires a non-empty 'name'.")
    model["name"] = model["name"].strip().lower()
    if model["name"] not in MODEL_REGISTRY:
        available = ", ".join(sorted(MODEL_REGISTRY))
        raise ValueError(f"Unknown model {model['name']!r}. Available models: {available}.")
    if not isinstance(model.get("pretrained", False), bool):
        raise ValueError("Model 'pretrained' must be true or false.")
    if "num_classes" in model and "num_labels" in model:
        if model["num_classes"] != model["num_labels"]:
            raise ValueError("Model 'num_classes' and 'num_labels' must match when both are set.")
    model["num_classes"] = model.pop("num_classes", model.pop("num_labels", None))
    if model["num_classes"] is not None:
        _positive_integer("model.num_classes", model["num_classes"])

    aliases = {
        "image_folder": "image_folder",
        "imagefolder": "image_folder",
        "csv": "csv_multilabel" if task == "multi_label" else "csv_classification",
        "csv_classification": "csv_classification",
        "classification_csv": "csv_classification",
        "csv_multilabel": "csv_multilabel",
        "multilabel_csv": "csv_multilabel",
    }
    requested_format = dataset.get("format")
    if requested_format is None:
        requested_format = "csv_multilabel" if task == "multi_label" else "image_folder"
    if not isinstance(requested_format, str) or requested_format.strip().lower() not in aliases:
        raise ValueError(
            "dataset.format must be image_folder, csv_classification, or csv_multilabel."
        )
    dataset_format = aliases[requested_format.strip().lower()]
    if task == "multi_label" and dataset_format != "csv_multilabel":
        raise ValueError("Multi-label classification requires dataset.format: csv_multilabel.")
    if task == "classification" and dataset_format == "csv_multilabel":
        raise ValueError("dataset.format csv_multilabel requires a multi_label task.")
    if dataset_format == "image_folder" and dataset.get("annotations_csv"):
        raise ValueError("dataset.annotations_csv can only be used with a CSV dataset format.")
    if dataset_format != "image_folder" and (
        dataset.get("train_dir") or dataset.get("validation_dir")
    ):
        raise ValueError("dataset.train_dir and validation_dir can only be used with image_folder.")
    label_columns = dataset.get("label_columns")
    if label_columns is not None:
        if (
            dataset_format != "csv_multilabel"
            or not isinstance(label_columns, list)
            or not label_columns
            or any(not isinstance(label, str) or not label for label in label_columns)
            or len(set(label_columns)) != len(label_columns)
        ):
            raise ValueError(
                "dataset.label_columns must be a unique, non-empty list for csv_multilabel."
            )
    dataset["format"] = dataset_format

    path_fields = ("path", "train_dir", "annotations_csv")
    if not any(dataset.get(field) for field in path_fields):
        raise ValueError("Dataset configuration requires 'path', 'train_dir', or 'annotations_csv'.")
    if dataset.get("validation_dir") and not dataset.get("train_dir"):
        raise ValueError("dataset.validation_dir requires dataset.train_dir.")
    for field in ("path", "train_dir", "validation_dir", "annotations_csv", "image_dir"):
        value = dataset.get(field)
        if value is not None and (not isinstance(value, str) or not value.strip()):
            raise ValueError(f"dataset.{field} must be a non-empty path string.")

    fraction = dataset.get("validation_fraction", 0.2)
    if isinstance(fraction, bool) or not isinstance(fraction, (int, float)) or not isfinite(fraction):
        raise ValueError("dataset.validation_fraction must be a finite number between 0 and 1.")
    if not 0 < fraction < 1:
        raise ValueError("dataset.validation_fraction must be greater than 0 and less than 1.")
    dataset["validation_fraction"] = float(fraction)

    image_size = dataset.get("image_size", 224)
    if isinstance(image_size, bool) or not (
        isinstance(image_size, int)
        or isinstance(image_size, list) and len(image_size) == 2
        and all(isinstance(value, int) and not isinstance(value, bool) for value in image_size)
    ):
        raise ValueError("dataset.image_size must be a positive integer or a pair of integers.")
    image_dimensions = [image_size] if isinstance(image_size, int) else image_size
    if any(value < 1 for value in image_dimensions):
        raise ValueError("dataset.image_size dimensions must be positive.")
    dataset["image_size"] = image_size
    for field, default in (("mean", [0.485, 0.456, 0.406]), ("std", [0.229, 0.224, 0.225])):
        values = dataset.get(field, default)
        if not isinstance(values, list) or len(values) != 3 or any(
            isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value)
            for value in values
        ):
            raise ValueError(f"dataset.{field} must contain three finite RGB values.")
        if field == "std" and any(value <= 0 for value in values):
            raise ValueError("dataset.std values must be greater than 0.")
        dataset[field] = values

    _non_negative_integer("training.seed", training.get("seed"))
    for field in ("epochs", "batch_size"):
        _positive_integer(f"training.{field}", training.get(field))
    _non_negative_integer("training.num_workers", training.get("num_workers"))
    learning_rate = training.get("learning_rate")
    if isinstance(learning_rate, bool) or not isinstance(learning_rate, (int, float)) or not isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("training.learning_rate must be finite and greater than 0.")
    training["learning_rate"] = float(learning_rate)
    device = training.get("device")
    if not isinstance(device, str) or device not in {"auto", "cpu", "cuda"}:
        raise ValueError("training.device must be auto, cpu, or cuda.")

    monitor = training.get("monitor", "val_loss")
    supported_monitors = {"val_loss", "val_accuracy"} if task == "classification" else {"val_loss", "val_micro_f1"}
    if not isinstance(monitor, str) or monitor not in supported_monitors:
        raise ValueError(f"training.monitor must be one of: {', '.join(sorted(supported_monitors))}.")
    expected_mode = "min" if monitor == "val_loss" else "max"
    mode = training.get("monitor_mode", expected_mode)
    if not isinstance(mode, str) or mode not in {"min", "max"} or mode != expected_mode:
        raise ValueError(f"training.monitor_mode must be '{expected_mode}' for {monitor}.")
    training["monitor_mode"] = mode
    training["amp"] = _validate_enabled_section("training.amp", training.get("amp", {"enabled": False}), {"enabled"})
    scheduler = _validate_enabled_section(
        "training.scheduler", training.get("scheduler", {"enabled": False}), {"enabled", "name", "step_size", "gamma"}
    )
    if scheduler.get("enabled"):
        if scheduler.get("name", "step_lr") != "step_lr":
            raise ValueError("Unsupported scheduler; currently supported: step_lr.")
        _positive_integer("training.scheduler.step_size", scheduler.get("step_size", 5))
        gamma = scheduler.get("gamma", 0.1)
        if isinstance(gamma, bool) or not isinstance(gamma, (int, float)) or not isfinite(gamma) or gamma <= 0:
            raise ValueError("training.scheduler.gamma must be finite and greater than 0.")
    training["scheduler"] = scheduler
    early = _validate_enabled_section(
        "training.early_stopping", training.get("early_stopping", {"enabled": False}),
        {"enabled", "patience", "min_delta", "mode"},
    )
    if early.get("enabled"):
        _positive_integer("training.early_stopping.patience", early.get("patience", 5))
        delta = early.get("min_delta", 0.0)
        if isinstance(delta, bool) or not isinstance(delta, (int, float)) or not isfinite(delta) or delta < 0:
            raise ValueError("training.early_stopping.min_delta must be finite and non-negative.")
        early_mode = early.get("mode", mode)
        if early_mode != mode:
            raise ValueError("training.early_stopping.mode must match training.monitor_mode.")
        early["mode"] = early_mode
    training["early_stopping"] = early

    threshold = evaluation.get("threshold", 0.5)
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("evaluation.threshold must be a finite number between 0 and 1.")
    primary_metric = evaluation.get("primary_metric", "accuracy")
    metrics = {"accuracy", "macro_f1", "macro_precision", "macro_recall"}
    if task == "multi_label":
        metrics.update({"micro_f1", "macro_auc", "micro_auc"})
    if not isinstance(primary_metric, str) or primary_metric not in metrics:
        raise ValueError(f"Unsupported evaluation.primary_metric for {task}: {primary_metric!r}.")
    evaluation["threshold"] = float(threshold)

    if not isinstance(output.get("directory"), str) or not output["directory"].strip():
        raise ValueError("output.directory must be a non-empty path string.")
    experiment_name = output.get("experiment_name")
    if experiment_name is not None:
        if not isinstance(experiment_name, str) or not experiment_name.strip():
            raise ValueError("output.experiment_name must be a non-empty name.")
        if any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-" for character in experiment_name):
            raise ValueError("output.experiment_name may contain only letters, numbers, '.', '_' and '-'.")

    return {
        "task": task_config,
        "dataset": dataset,
        "model": model,
        "training": training,
        "evaluation": evaluation,
        "output": output,
    }


def _check_keys(section: str, values: Mapping[str, Any], allowed: set[str]) -> None:
    unknown = sorted(set(values).difference(allowed))
    if unknown:
        raise ValueError(f"Unknown option in configuration section '{section}': {unknown[0]}.")


def _positive_integer(name: str, value: Any) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer.")


def _non_negative_integer(name: str, value: Any) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer.")


def _validate_enabled_section(name: str, value: Any, allowed: set[str]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping.")
    result = dict(value)
    _check_keys(name, result, allowed)
    if not isinstance(result.get("enabled", False), bool):
        raise ValueError(f"{name}.enabled must be true or false.")
    return result
