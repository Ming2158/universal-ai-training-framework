"""A small registry for image classification model architectures."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from torch import nn
from torchvision import models

ModelBuilder = Callable[[int, bool], nn.Module]


def _build_resnet50(output_count: int, pretrained: bool) -> nn.Module:
    weights = models.ResNet50_Weights.DEFAULT if pretrained else None
    model = models.resnet50(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, output_count)
    return model


def _build_convnext_tiny(output_count: int, pretrained: bool) -> nn.Module:
    weights = models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
    model = models.convnext_tiny(weights=weights)
    classifier = model.classifier[-1]
    model.classifier[-1] = nn.Linear(classifier.in_features, output_count)
    return model


def _build_efficientnet_b0(output_count: int, pretrained: bool) -> nn.Module:
    weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
    model = models.efficientnet_b0(weights=weights)
    classifier = model.classifier[-1]
    model.classifier[-1] = nn.Linear(classifier.in_features, output_count)
    return model


def _build_vit_b_16(output_count: int, pretrained: bool) -> nn.Module:
    weights = models.ViT_B_16_Weights.DEFAULT if pretrained else None
    model = models.vit_b_16(weights=weights)
    classifier = model.heads.head
    model.heads.head = nn.Linear(classifier.in_features, output_count)
    return model


MODEL_REGISTRY: dict[str, ModelBuilder] = {
    "resnet50": _build_resnet50,
    "convnext": _build_convnext_tiny,
    "efficientnet": _build_efficientnet_b0,
    "vit": _build_vit_b_16,
}

_TASK_ALIASES = {
    "classification": "classification",
    "single_label": "classification",
    "image_classification": "classification",
    "multi_label": "multi_label",
    "multilabel": "multi_label",
    "multi_label_classification": "multi_label",
    "multilabel_classification": "multi_label",
}


def normalize_task(task: str) -> str:
    if not isinstance(task, str):
        raise ValueError("Task name must be a string.")
    normalized = task.strip().lower().replace("-", "_")
    try:
        return _TASK_ALIASES[normalized]
    except KeyError as error:
        supported = "classification, multi_label"
        raise ValueError(f"Unsupported task {task!r}; expected one of: {supported}.") from error


def register_model(name: str, builder: ModelBuilder) -> None:
    """Register a builder that accepts (output_count, pretrained)."""
    normalized = name.strip().lower()
    if not normalized:
        raise ValueError("Model name cannot be empty.")
    if normalized in MODEL_REGISTRY:
        raise ValueError(f"Model is already registered: {normalized}")
    if not callable(builder):
        raise TypeError("Model builder must be callable.")
    MODEL_REGISTRY[normalized] = builder


def get_model(
    name: str,
    num_classes: int,
    pretrained: bool = False,
    *,
    task: str = "classification",
) -> nn.Module:
    """Create a model with a fresh output head sized for the selected task.

    For multi-label tasks, ``num_classes`` is the number of independent labels.
    The returned model emits raw logits and does not apply softmax or sigmoid.
    """
    if not isinstance(name, str):
        raise ValueError("Model name must be a string.")
    if isinstance(num_classes, bool) or not isinstance(num_classes, int) or num_classes < 1:
        raise ValueError("num_classes must be a positive integer.")
    if not isinstance(pretrained, bool):
        raise ValueError("pretrained must be a boolean.")
    normalize_task(task)

    normalized_name = name.strip().lower()
    try:
        builder = MODEL_REGISTRY[normalized_name]
    except KeyError as error:
        available = ", ".join(sorted(MODEL_REGISTRY))
        raise ValueError(
            f"Unknown model {name!r}. Available models: {available}."
        ) from error
    return builder(num_classes, pretrained)


def build_model_from_config(configuration: Mapping[str, Any]) -> nn.Module:
    """Build a model from the top-level ``task`` and ``model`` config sections."""
    model_config = configuration.get("model", configuration)
    if not isinstance(model_config, Mapping):
        raise ValueError("Configuration section 'model' must be a mapping.")

    task_config = configuration.get("task", {})
    if isinstance(task_config, str):
        task = task_config
    elif isinstance(task_config, Mapping):
        task = task_config.get("type", "classification")
    else:
        raise ValueError("Configuration section 'task' must be a mapping or task name.")
    if not isinstance(task, str):
        raise ValueError("Configuration task type must be a string.")

    output_count = model_config.get("num_classes", model_config.get("num_labels"))
    if output_count is None:
        raise ValueError("Model configuration is missing 'num_classes' (or 'num_labels').")
    name = model_config.get("name")
    if not isinstance(name, str):
        raise ValueError("Model configuration is missing a string 'name'.")

    return get_model(
        name=name,
        num_classes=output_count,
        pretrained=model_config.get("pretrained", False),
        task=task,
    )
