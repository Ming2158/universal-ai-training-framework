"""YAML loading, defaults, and early validation for training configuration."""

from __future__ import annotations

import os
from collections.abc import Mapping
from copy import deepcopy
from math import isfinite
from pathlib import Path
from typing import Any

import yaml


class ConfigurationError(ValueError):
    """Raised when a UATF configuration is invalid or cannot be resolved."""


DEFAULT_CONFIG: dict[str, Any] = {
    "task": {"type": "classification"},
    "dataset": {
        "validation_fraction": 0.2,
        "image_size": 224,
        "resize_enabled": True,
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
        "augmentation": {
            "enabled": False,
            "horizontal_flip_probability": 0.5,
            "rotation_degrees": 0.0,
            "brightness": 0.0,
            "contrast": 0.0,
            "saturation": 0.0,
            "hue": 0.0,
        },
    },
    "model": {"name": "resnet50", "pretrained": False},
    "training": {
        "seed": 42,
        "epochs": 10,
        "batch_size": 32,
        "learning_rate": 0.001,
        "num_workers": 0,
        "device": "auto",
        "loss": {"name": "auto"},
        "optimizer": {"name": "adam", "weight_decay": 0.0},
        "gradient_accumulation_steps": 1,
        "gradient_clip_norm": None,
        "resume_from": None,
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
    "output": {
        "directory": "outputs",
        "save_predictions": True,
        "save_plots": True,
    },
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
        raise ConfigurationError(f"Invalid YAML in configuration {config_path}: {error}") from error

    if not isinstance(configuration, dict):
        raise ConfigurationError(f"Configuration must contain a YAML mapping: {config_path}")
    return configuration


def dump_config(configuration: Mapping[str, Any], path: str | Path) -> Path:
    """Write a YAML mapping in a readable, stable field order."""
    destination = Path(path).expanduser()
    try:
        with destination.open("w", encoding="utf-8", newline="\n") as output:
            yaml.safe_dump(
                dict(configuration),
                output,
                sort_keys=False,
                allow_unicode=True,
                default_flow_style=False,
            )
    except (OSError, yaml.YAMLError) as error:
        raise ConfigurationError(
            f"Could not write configuration to {destination}: {error}"
        ) from error
    return destination


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


def resolve_config(
    configuration: Mapping[str, Any],
    overrides: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Apply explicit overrides over supplied values, then defaults and validation."""
    if not isinstance(configuration, Mapping):
        raise ConfigurationError("Configuration must be a mapping.")
    if overrides is not None and not isinstance(overrides, Mapping):
        raise ConfigurationError("Configuration overrides must be a mapping.")
    supplied = merge_config(configuration, overrides or {})
    return validate_config(supplied)


def validate_config(configuration: Mapping[str, Any]) -> dict[str, Any]:
    """Validate known options and return a normalized, complete configuration."""
    from uatf.evaluation.metrics import metric_names_for_task, training_metric_names_for_task
    from uatf.models.factory import MODEL_REGISTRY, get_model_adapter, normalize_task
    from uatf.models.losses import get_loss_adapter

    if not isinstance(configuration, Mapping):
        raise ConfigurationError("Configuration must be a mapping.")
    config = merge_config(DEFAULT_CONFIG, configuration)
    allowed_sections = {"task", "dataset", "model", "training", "evaluation", "output"}
    unknown_sections = sorted(set(config).difference(allowed_sections))
    if unknown_sections:
        raise ConfigurationError(f"Unknown configuration section: {unknown_sections[0]}.")

    for section in allowed_sections:
        if not isinstance(config.get(section), Mapping):
            raise ConfigurationError(f"Configuration section '{section}' must be a mapping.")

    raw_task = configuration.get("task", {})
    task_config = dict(config["task"])
    dataset = dict(config["dataset"])
    dataset_task = dataset.pop("type", None)
    requested_task = raw_task.get("type") if isinstance(raw_task, Mapping) else None
    try:
        if dataset_task is not None and requested_task is not None:
            if normalize_task(dataset_task) != normalize_task(requested_task):
                raise ConfigurationError("dataset.type and task.type specify different task types.")
    except ValueError as error:
        raise ConfigurationError(str(error)) from error
    task_value = dataset_task if dataset_task is not None else requested_task
    if task_value is None:
        task_value = task_config.get("type", "classification")
    try:
        task = normalize_task(task_value)
    except ValueError as error:
        raise ConfigurationError(str(error)) from error
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
            "label_columns", "validation_fraction", "image_size", "resize_enabled", "mean", "std", "augmentation",
        },
    )
    _check_keys("model", model, {"name", "pretrained", "num_classes", "num_labels"})
    _check_keys(
        "training",
        training,
        {
            "seed", "epochs", "batch_size", "learning_rate", "num_workers", "device",
            "loss", "optimizer", "gradient_accumulation_steps", "gradient_clip_norm",
            "resume_from", "monitor", "monitor_mode", "amp", "scheduler", "early_stopping",
        },
    )
    _check_keys("evaluation", evaluation, {"primary_metric", "threshold"})
    _check_keys(
        "output",
        output,
        {"directory", "experiment_name", "save_predictions", "save_plots"},
    )

    if not isinstance(model.get("name"), str) or not model["name"].strip():
        raise ConfigurationError("Configuration section 'model' requires a non-empty 'name'.")
    model["name"] = model["name"].strip().lower()
    if model["name"] not in MODEL_REGISTRY:
        available = ", ".join(sorted(MODEL_REGISTRY))
        raise ConfigurationError(f"Unknown model {model['name']!r}. Available models: {available}.")
    if not isinstance(model.get("pretrained", False), bool):
        raise ConfigurationError("Model 'pretrained' must be true or false.")
    adapter = get_model_adapter(model["name"])
    if task not in adapter.supported_tasks:
        raise ConfigurationError(
            f"Model {model['name']!r} does not support task {task!r}; "
            f"supported tasks: {', '.join(adapter.supported_tasks)}."
        )
    if model.get("pretrained", False) and not adapter.supports_pretrained:
        raise ConfigurationError(
            f"Model {model['name']!r} does not support pretrained weights."
        )
    if "num_classes" in model and "num_labels" in model:
        if model["num_classes"] != model["num_labels"]:
            raise ConfigurationError(
                "Model 'num_classes' and 'num_labels' must match when both are set."
            )
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
        raise ConfigurationError(
            "dataset.format must be image_folder, csv_classification, or csv_multilabel."
        )
    dataset_format = aliases[requested_format.strip().lower()]
    if task == "multi_label" and dataset_format != "csv_multilabel":
        raise ConfigurationError("Multi-label classification requires dataset.format: csv_multilabel.")
    if task == "classification" and dataset_format == "csv_multilabel":
        raise ConfigurationError("dataset.format csv_multilabel requires a multi_label task.")
    if dataset_format == "image_folder" and dataset.get("annotations_csv"):
        raise ConfigurationError("dataset.annotations_csv can only be used with a CSV dataset format.")
    if dataset_format != "image_folder" and (
        dataset.get("train_dir") or dataset.get("validation_dir")
    ):
        raise ConfigurationError("dataset.train_dir and validation_dir can only be used with image_folder.")
    label_columns = dataset.get("label_columns")
    if label_columns is not None:
        if (
            dataset_format != "csv_multilabel"
            or not isinstance(label_columns, list)
            or not label_columns
            or any(not isinstance(label, str) or not label for label in label_columns)
            or len(set(label_columns)) != len(label_columns)
        ):
            raise ConfigurationError(
                "dataset.label_columns must be a unique, non-empty list for csv_multilabel."
            )
    dataset["format"] = dataset_format

    path_fields = ("path", "train_dir", "annotations_csv")
    if not any(dataset.get(field) for field in path_fields):
        raise ConfigurationError(
            "Dataset configuration requires 'path', 'train_dir', or 'annotations_csv'."
        )
    if dataset.get("validation_dir") and not dataset.get("train_dir"):
        raise ConfigurationError("dataset.validation_dir requires dataset.train_dir.")
    for field in ("path", "train_dir", "validation_dir", "annotations_csv", "image_dir"):
        value = dataset.get(field)
        if value is not None and (not isinstance(value, str) or not value.strip()):
            raise ConfigurationError(f"dataset.{field} must be a non-empty path string.")

    fraction = dataset.get("validation_fraction", 0.2)
    if isinstance(fraction, bool) or not isinstance(fraction, (int, float)) or not isfinite(fraction):
        raise ConfigurationError("dataset.validation_fraction must be a finite number between 0 and 1.")
    if not 0 < fraction < 1:
        raise ConfigurationError("dataset.validation_fraction must be greater than 0 and less than 1.")
    dataset["validation_fraction"] = float(fraction)

    image_size = dataset.get("image_size", 224)
    if isinstance(image_size, bool) or not (
        isinstance(image_size, int)
        or isinstance(image_size, list) and len(image_size) == 2
        and all(isinstance(value, int) and not isinstance(value, bool) for value in image_size)
    ):
        raise ConfigurationError("dataset.image_size must be a positive integer or a pair of integers.")
    image_dimensions = [image_size] if isinstance(image_size, int) else image_size
    if any(value < 1 for value in image_dimensions):
        raise ConfigurationError("dataset.image_size dimensions must be positive.")
    dataset["image_size"] = image_size
    resize_enabled = dataset.get("resize_enabled", True)
    if not isinstance(resize_enabled, bool):
        raise ConfigurationError("dataset.resize_enabled must be a boolean.")
    dataset["resize_enabled"] = resize_enabled
    if adapter.input_size_required and not resize_enabled:
        raise ConfigurationError(
            f"Model {model['name']!r} requires fixed-size image resizing."
        )
    for field, default in (("mean", [0.485, 0.456, 0.406]), ("std", [0.229, 0.224, 0.225])):
        values = dataset.get(field, default)
        if not isinstance(values, list) or len(values) != 3 or any(
            isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value)
            for value in values
        ):
            raise ConfigurationError(f"dataset.{field} must contain three finite RGB values.")
        if field == "std" and any(value <= 0 for value in values):
            raise ConfigurationError("dataset.std values must be greater than 0.")
        dataset[field] = values

    if adapter.input_size_required:
        actual_size = (image_size, image_size) if isinstance(image_size, int) else tuple(image_size)
        if resize_enabled and actual_size != adapter.input_size:
            required = " x ".join(str(value) for value in adapter.input_size or ())
            raise ConfigurationError(
                f"Model {model['name']!r} requires dataset.image_size {required}; "
                f"received {actual_size}."
            )

    augmentation = dataset.get("augmentation")
    if not isinstance(augmentation, Mapping):
        raise ConfigurationError("dataset.augmentation must be a mapping.")
    augmentation = dict(augmentation)
    _check_keys(
        "dataset.augmentation",
        augmentation,
        {
            "enabled", "horizontal_flip_probability", "rotation_degrees", "brightness",
            "contrast", "saturation", "hue",
        },
    )
    if not isinstance(augmentation.get("enabled", False), bool):
        raise ConfigurationError("dataset.augmentation.enabled must be true or false.")
    for field, default in (
        ("horizontal_flip_probability", 0.5),
        ("rotation_degrees", 0.0),
        ("brightness", 0.0),
        ("contrast", 0.0),
        ("saturation", 0.0),
        ("hue", 0.0),
    ):
        value = augmentation.get(field, default)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value):
            raise ConfigurationError(f"dataset.augmentation.{field} must be finite.")
        if value < 0:
            raise ConfigurationError(f"dataset.augmentation.{field} must be non-negative.")
        augmentation[field] = float(value)
    if augmentation["horizontal_flip_probability"] > 1:
        raise ConfigurationError(
            "dataset.augmentation.horizontal_flip_probability must be between 0 and 1."
        )
    if augmentation["rotation_degrees"] > 180:
        raise ConfigurationError("dataset.augmentation.rotation_degrees cannot exceed 180.")
    if augmentation["hue"] > 0.5:
        raise ConfigurationError("dataset.augmentation.hue cannot exceed 0.5.")
    dataset["augmentation"] = augmentation

    _non_negative_integer("training.seed", training.get("seed"))
    for field in ("epochs", "batch_size"):
        _positive_integer(f"training.{field}", training.get(field))
    _non_negative_integer("training.num_workers", training.get("num_workers"))
    learning_rate = training.get("learning_rate")
    if isinstance(learning_rate, bool) or not isinstance(learning_rate, (int, float)) or not isfinite(learning_rate) or learning_rate <= 0:
        raise ConfigurationError("training.learning_rate must be finite and greater than 0.")
    training["learning_rate"] = float(learning_rate)
    device = training.get("device")
    if not isinstance(device, str) or device not in {"auto", "cpu", "cuda"}:
        raise ConfigurationError("training.device must be auto, cpu, or cuda.")

    loss_config = training.get("loss")
    if not isinstance(loss_config, Mapping):
        raise ConfigurationError("training.loss must be a mapping.")
    loss_config = dict(loss_config)
    _check_keys("training.loss", loss_config, {"name"})
    loss_name = loss_config.get("name", "auto")
    if not isinstance(loss_name, str) or not loss_name.strip():
        raise ConfigurationError("training.loss.name must be a non-empty string.")
    try:
        loss_adapter = get_loss_adapter(loss_name, task)
    except ValueError as error:
        raise ConfigurationError(str(error)) from error
    training["loss"] = {"name": loss_adapter.identifier}

    accumulation_steps = training.get("gradient_accumulation_steps", 1)
    _positive_integer("training.gradient_accumulation_steps", accumulation_steps)
    clip_norm = training.get("gradient_clip_norm")
    if clip_norm is not None:
        if isinstance(clip_norm, bool) or not isinstance(clip_norm, (int, float)) or not isfinite(clip_norm) or clip_norm <= 0:
            raise ConfigurationError(
                "training.gradient_clip_norm must be null or a finite number greater than 0."
            )
        training["gradient_clip_norm"] = float(clip_norm)

    optimizer = training.get("optimizer")
    if not isinstance(optimizer, Mapping):
        raise ConfigurationError("training.optimizer must be a mapping.")
    optimizer = dict(optimizer)
    optimizer_name = optimizer.get("name", "adam")
    if not isinstance(optimizer_name, str) or optimizer_name.strip().lower() not in {
        "adam", "adamw", "sgd"
    }:
        raise ConfigurationError("training.optimizer.name must be 'adam', 'adamw', or 'sgd'.")
    optimizer_name = optimizer_name.strip().lower()
    optimizer["name"] = optimizer_name
    optimizer_keys = {
        "adam": {"name", "weight_decay", "betas", "eps"},
        "adamw": {"name", "weight_decay", "betas", "eps"},
        "sgd": {"name", "weight_decay", "momentum", "nesterov"},
    }[optimizer_name]
    _check_keys("training.optimizer", optimizer, optimizer_keys)
    weight_decay = optimizer.get("weight_decay", 0.0)
    if (
        isinstance(weight_decay, bool)
        or not isinstance(weight_decay, (int, float))
        or not isfinite(weight_decay)
        or weight_decay < 0
    ):
        raise ConfigurationError("training.optimizer.weight_decay must be finite and non-negative.")
    optimizer["weight_decay"] = float(weight_decay)
    if optimizer_name in {"adam", "adamw"}:
        betas = optimizer.get("betas", [0.9, 0.999])
        if (
            not isinstance(betas, list)
            or len(betas) != 2
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not isfinite(value)
                or not 0 <= value < 1
                for value in betas
            )
        ):
            raise ConfigurationError(
                "training.optimizer.betas must contain two finite values in [0, 1)."
            )
        optimizer["betas"] = [float(value) for value in betas]
        epsilon = optimizer.get("eps", 1e-8)
        if (
            isinstance(epsilon, bool)
            or not isinstance(epsilon, (int, float))
            or not isfinite(epsilon)
            or epsilon <= 0
        ):
            raise ConfigurationError("training.optimizer.eps must be finite and greater than 0.")
        optimizer["eps"] = float(epsilon)
    else:
        momentum = optimizer.get("momentum", 0.0)
        if (
            isinstance(momentum, bool)
            or not isinstance(momentum, (int, float))
            or not isfinite(momentum)
            or not 0 <= momentum < 1
        ):
            raise ConfigurationError(
                "training.optimizer.momentum must be finite and in the range [0, 1)."
            )
        optimizer["momentum"] = float(momentum)
        nesterov = optimizer.get("nesterov", False)
        if not isinstance(nesterov, bool):
            raise ConfigurationError("training.optimizer.nesterov must be true or false.")
        if nesterov and momentum <= 0:
            raise ConfigurationError(
                "training.optimizer.nesterov requires training.optimizer.momentum greater than 0."
            )
        optimizer["nesterov"] = nesterov
    training["optimizer"] = optimizer

    resume_from = training.get("resume_from")
    if resume_from is not None:
        if not isinstance(resume_from, str) or not resume_from.strip():
            raise ConfigurationError("training.resume_from must be a checkpoint file path.")
        checkpoint_path = Path(resume_from).expanduser()
        if not checkpoint_path.is_file():
            raise ConfigurationError(
                f"training.resume_from checkpoint does not exist or is not a file: "
                f"{checkpoint_path}. Set it to a valid .pth checkpoint or null."
            )
        training["resume_from"] = str(checkpoint_path.resolve())

    monitor = training.get("monitor", "val_loss")
    supported_monitors = {"val_loss", "val_accuracy"} if task == "classification" else {"val_loss", "val_micro_f1"}
    if not isinstance(monitor, str) or monitor not in supported_monitors:
        raise ConfigurationError(
            f"training.monitor must be one of: {', '.join(sorted(supported_monitors))}."
        )
    expected_mode = "min" if monitor == "val_loss" else "max"
    mode = training.get("monitor_mode", expected_mode)
    if not isinstance(mode, str) or mode not in {"min", "max"} or mode != expected_mode:
        raise ConfigurationError(f"training.monitor_mode must be '{expected_mode}' for {monitor}.")
    training["monitor_mode"] = mode
    training["amp"] = _validate_enabled_section("training.amp", training.get("amp", {"enabled": False}), {"enabled"})
    if training["amp"].get("enabled") and device == "cpu":
        raise ConfigurationError(
            "training.amp.enabled cannot be true when training.device is 'cpu'; "
            "select cuda or disable AMP."
        )
    scheduler = _validate_enabled_section(
        "training.scheduler",
        training.get("scheduler", {"enabled": False}),
        {"enabled", "name", "step_size", "gamma", "t_max", "eta_min"},
    )
    scheduler_name = scheduler.get("name", "step_lr")
    if scheduler_name == "step_lr":
        _positive_integer("training.scheduler.step_size", scheduler.get("step_size", 5))
        gamma = scheduler.get("gamma", 0.1)
        if isinstance(gamma, bool) or not isinstance(gamma, (int, float)) or not isfinite(gamma) or gamma <= 0:
            raise ConfigurationError("training.scheduler.gamma must be finite and greater than 0.")
        scheduler.update({"name": scheduler_name, "step_size": int(scheduler.get("step_size", 5)), "gamma": float(gamma)})
    elif scheduler_name == "cosine":
        t_max = scheduler.get("t_max", training["epochs"])
        _positive_integer("training.scheduler.t_max", t_max)
        eta_min = scheduler.get("eta_min", 0.0)
        if isinstance(eta_min, bool) or not isinstance(eta_min, (int, float)) or not isfinite(eta_min) or eta_min < 0 or eta_min > training["learning_rate"]:
            raise ConfigurationError(
                "training.scheduler.eta_min must be finite, non-negative, and no greater than learning_rate."
            )
        scheduler.update({"name": scheduler_name, "t_max": t_max, "eta_min": float(eta_min)})
    else:
        raise ConfigurationError("Unsupported scheduler; currently supported: step_lr, cosine.")
    training["scheduler"] = scheduler
    early = _validate_enabled_section(
        "training.early_stopping", training.get("early_stopping", {"enabled": False}),
        {"enabled", "patience", "min_delta", "mode"},
    )
    _positive_integer("training.early_stopping.patience", early.get("patience", 5))
    delta = early.get("min_delta", 0.0)
    if isinstance(delta, bool) or not isinstance(delta, (int, float)) or not isfinite(delta) or delta < 0:
        raise ConfigurationError("training.early_stopping.min_delta must be finite and non-negative.")
    early_mode = early.get("mode", mode)
    if early_mode != mode:
        raise ConfigurationError("training.early_stopping.mode must match training.monitor_mode.")
    early["mode"] = early_mode
    training["early_stopping"] = early

    threshold = evaluation.get("threshold", 0.5)
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not isfinite(threshold) or not 0 <= threshold <= 1:
        raise ConfigurationError("evaluation.threshold must be a finite number between 0 and 1.")
    if "primary_metric" not in configuration.get("evaluation", {}):
        evaluation["primary_metric"] = "micro_f1" if task == "multi_label" else "accuracy"
    primary_metric = evaluation.get("primary_metric")
    metrics = set(metric_names_for_task(task))
    if not isinstance(primary_metric, str) or primary_metric not in metrics:
        available_metrics = ", ".join(sorted(metrics))
        raise ConfigurationError(
            f"Unsupported evaluation.primary_metric for {task}: {primary_metric!r}. "
            f"Choose one of: {available_metrics}."
        )
    evaluation["threshold"] = float(threshold)

    if not isinstance(output.get("directory"), str) or not output["directory"].strip():
        raise ConfigurationError("output.directory must be a non-empty path string.")
    _validate_output_directory(output["directory"])
    for field in ("save_predictions", "save_plots"):
        if not isinstance(output.get(field), bool):
            raise ConfigurationError(f"output.{field} must be true or false.")
    experiment_name = output.get("experiment_name")
    if experiment_name is not None:
        if not isinstance(experiment_name, str) or not experiment_name.strip():
            raise ConfigurationError("output.experiment_name must be a non-empty name.")
        if any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-" for character in experiment_name):
            raise ConfigurationError(
                "output.experiment_name may contain only letters, numbers, '.', '_' and '-'."
            )

    monitor_name = training.get("monitor", "val_loss").removeprefix("val_")
    supported_monitors = {"loss", *training_metric_names_for_task(task)}
    if monitor_name not in supported_monitors:
        raise ConfigurationError(
            f"training.monitor is not available for task {task!r}; choose val_loss or one of: "
            f"{', '.join(f'val_{name}' for name in training_metric_names_for_task(task))}."
        )

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
        raise ConfigurationError(
            f"Unknown option in configuration section '{section}': {unknown[0]}."
        )


def _positive_integer(name: str, value: Any) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ConfigurationError(f"{name} must be a positive integer.")


def _non_negative_integer(name: str, value: Any) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ConfigurationError(f"{name} must be a non-negative integer.")


def _validate_enabled_section(name: str, value: Any, allowed: set[str]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigurationError(f"{name} must be a mapping.")
    result = dict(value)
    _check_keys(name, result, allowed)
    if not isinstance(result.get("enabled", False), bool):
        raise ConfigurationError(f"{name}.enabled must be true or false.")
    return result


def _validate_output_directory(directory: str) -> None:
    try:
        target = Path(directory).expanduser().resolve()
    except (OSError, RuntimeError, ValueError) as error:
        raise ConfigurationError(f"output.directory is not a valid path: {directory!r}.") from error
    if target.exists() and not target.is_dir():
        raise ConfigurationError(
            f"output.directory points to a file: {target}. Choose a directory path."
        )

    writable_ancestor = target
    while not writable_ancestor.exists() and writable_ancestor != writable_ancestor.parent:
        writable_ancestor = writable_ancestor.parent
    if not writable_ancestor.is_dir():
        raise ConfigurationError(
            f"output.directory has no existing parent directory: {writable_ancestor}."
        )
    if not os.access(writable_ancestor, os.W_OK):
        raise ConfigurationError(
            f"output.directory is not writable through {writable_ancestor}. "
            "Choose a writable directory or update its permissions."
        )
