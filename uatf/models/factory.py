"""Registered image-classification model builders and their capabilities."""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torchvision import models

ModelBuilder = Callable[[int, bool], nn.Module]


@dataclass(frozen=True)
class ModelAdapter:
    """Public model contract used by configuration, UI, training, and inference."""

    identifier: str
    display_name: str
    builder: ModelBuilder
    supported_tasks: tuple[str, ...]
    supports_pretrained: bool
    pretrained_source: str | None
    pretrained_url: str | None
    input_size: tuple[int, int] | None
    input_size_required: bool
    normalization_mean: tuple[float, float, float]
    normalization_std: tuple[float, float, float]
    classifier_head: str
    checkpoint_format: str
    inference_config: tuple[str, ...]
    limitations: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return serializable capability metadata without the Python builder."""
        return {
            "identifier": self.identifier,
            "display_name": self.display_name,
            "supported_tasks": list(self.supported_tasks),
            "supports_pretrained": self.supports_pretrained,
            "pretrained_source": self.pretrained_source,
            "pretrained_url": self.pretrained_url,
            "pretrained_cache_path": (
                str(self.pretrained_cache_path) if self.pretrained_cache_path else None
            ),
            "input_size": list(self.input_size) if self.input_size else None,
            "input_size_required": self.input_size_required,
            "normalization_mean": list(self.normalization_mean),
            "normalization_std": list(self.normalization_std),
            "classifier_head": self.classifier_head,
            "checkpoint_format": self.checkpoint_format,
            "inference_config": list(self.inference_config),
            "limitations": list(self.limitations),
        }

    @property
    def pretrained_cache_path(self) -> Path | None:
        if not self.pretrained_url:
            return None
        filename = self.pretrained_url.rsplit("/", 1)[-1]
        return Path(torch.hub.get_dir()) / "checkpoints" / filename


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


_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def _adapter(
    identifier: str,
    display_name: str,
    builder: ModelBuilder,
    *,
    weight_source: str,
    weight_url: str,
    classifier_head: str,
    input_size: tuple[int, int] | None = (224, 224),
    input_size_required: bool = False,
    limitations: tuple[str, ...] = (),
) -> ModelAdapter:
    return ModelAdapter(
        identifier=identifier,
        display_name=display_name,
        builder=builder,
        supported_tasks=("classification", "multi_label"),
        supports_pretrained=True,
        pretrained_source=weight_source,
        pretrained_url=weight_url,
        input_size=input_size,
        input_size_required=input_size_required,
        normalization_mean=_IMAGENET_MEAN,
        normalization_std=_IMAGENET_STD,
        classifier_head=classifier_head,
        checkpoint_format="PyTorch state_dict and UATF training checkpoint",
        inference_config=(
            "dataset.image_size",
            "dataset.resize_enabled",
            "dataset.mean",
            "dataset.std",
            "evaluation.threshold",
        ),
        limitations=limitations,
    )


MODEL_REGISTRY: dict[str, ModelBuilder] = {
    "resnet50": _build_resnet50,
    "convnext": _build_convnext_tiny,
    "efficientnet": _build_efficientnet_b0,
    "vit": _build_vit_b_16,
}

MODEL_ADAPTERS: dict[str, ModelAdapter] = {
    "resnet50": _adapter(
        "resnet50", "ResNet-50", _build_resnet50,
        weight_source="torchvision ResNet50_Weights.DEFAULT",
        weight_url=models.ResNet50_Weights.DEFAULT.url,
        classifier_head="fc",
    ),
    "convnext": _adapter(
        "convnext", "ConvNeXt-Tiny", _build_convnext_tiny,
        weight_source="torchvision ConvNeXt_Tiny_Weights.DEFAULT",
        weight_url=models.ConvNeXt_Tiny_Weights.DEFAULT.url,
        classifier_head="classifier[-1]",
    ),
    "efficientnet": _adapter(
        "efficientnet", "EfficientNet-B0", _build_efficientnet_b0,
        weight_source="torchvision EfficientNet_B0_Weights.DEFAULT",
        weight_url=models.EfficientNet_B0_Weights.DEFAULT.url,
        classifier_head="classifier[-1]",
    ),
    "vit": _adapter(
        "vit", "ViT-B/16", _build_vit_b_16,
        weight_source="torchvision ViT_B_16_Weights.DEFAULT",
        weight_url=models.ViT_B_16_Weights.DEFAULT.url,
        classifier_head="heads.head",
        input_size_required=True,
        limitations=("Requires 224 x 224 input images.",),
    ),
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


def register_model(
    name: str,
    builder: ModelBuilder,
    *,
    display_name: str | None = None,
    supported_tasks: Sequence[str] = ("classification", "multi_label"),
    supports_pretrained: bool = True,
    pretrained_source: str | None = None,
    pretrained_url: str | None = None,
    input_size: tuple[int, int] | None = None,
    input_size_required: bool = False,
    normalization_mean: Sequence[float] = _IMAGENET_MEAN,
    normalization_std: Sequence[float] = _IMAGENET_STD,
    classifier_head: str = "builder-defined",
    limitations: Sequence[str] = (),
) -> None:
    """Register the legacy builder plus its declared task and inference contract."""
    if not isinstance(name, str):
        raise ValueError("Model name must be a string.")
    normalized = name.strip().lower()
    if not normalized:
        raise ValueError("Model name cannot be empty.")
    if normalized in MODEL_REGISTRY:
        raise ValueError(f"Model is already registered: {normalized}")
    if not callable(builder):
        raise TypeError("Model builder must be callable.")
    if not isinstance(supports_pretrained, bool):
        raise ValueError("supports_pretrained must be a boolean.")
    if not isinstance(input_size_required, bool):
        raise ValueError("input_size_required must be a boolean.")
    tasks = tuple(dict.fromkeys(normalize_task(task) for task in supported_tasks))
    if not tasks:
        raise ValueError("A model adapter must support at least one task.")
    if len(normalization_mean) != 3 or len(normalization_std) != 3:
        raise ValueError("Model normalization mean and std must contain three RGB values.")
    if any(value <= 0 for value in normalization_std):
        raise ValueError("Model normalization standard deviations must be positive.")
    if input_size is not None and (
        len(input_size) != 2
        or any(
            isinstance(size, bool) or not isinstance(size, int) or size < 1
            for size in input_size
        )
    ):
        raise ValueError("Model input_size must be a pair of positive integers or None.")
    if input_size_required and input_size is None:
        raise ValueError("A required model input size must be specified.")

    adapter = ModelAdapter(
        identifier=normalized,
        display_name=display_name.strip() if isinstance(display_name, str) and display_name.strip() else normalized,
        builder=builder,
        supported_tasks=tasks,
        supports_pretrained=supports_pretrained,
        pretrained_source=pretrained_source,
        pretrained_url=pretrained_url,
        input_size=input_size,
        input_size_required=input_size_required,
        normalization_mean=tuple(float(value) for value in normalization_mean),
        normalization_std=tuple(float(value) for value in normalization_std),
        classifier_head=classifier_head,
        checkpoint_format="PyTorch state_dict and UATF training checkpoint",
        inference_config=("dataset.image_size", "dataset.mean", "dataset.std", "evaluation.threshold"),
        limitations=tuple(str(value) for value in limitations),
    )
    MODEL_REGISTRY[normalized] = builder
    MODEL_ADAPTERS[normalized] = adapter


def get_model_adapter(name: str) -> ModelAdapter:
    if not isinstance(name, str):
        raise ValueError("Model name must be a string.")
    normalized_name = name.strip().lower()
    if normalized_name not in MODEL_REGISTRY:
        available = ", ".join(sorted(MODEL_REGISTRY))
        raise ValueError(f"Unknown model {name!r}. Available models: {available}.")
    adapter = MODEL_ADAPTERS.get(normalized_name)
    if adapter is not None and adapter.builder is MODEL_REGISTRY[normalized_name]:
        return adapter
    # Direct dictionary registration remains supported for older extensions.
    return ModelAdapter(
        identifier=normalized_name,
        display_name=normalized_name,
        builder=MODEL_REGISTRY[normalized_name],
        supported_tasks=("classification", "multi_label"),
        supports_pretrained=True,
        pretrained_source=None,
        pretrained_url=None,
        input_size=None,
        input_size_required=False,
        normalization_mean=_IMAGENET_MEAN,
        normalization_std=_IMAGENET_STD,
        classifier_head="legacy builder-defined",
        checkpoint_format="PyTorch state_dict and UATF training checkpoint",
        inference_config=("dataset.image_size", "dataset.mean", "dataset.std", "evaluation.threshold"),
        limitations=("Registered through the legacy builder map; capabilities were not declared.",),
    )


def get_model_adapters() -> tuple[ModelAdapter, ...]:
    """Return adapters in stable identifier order for UI and documentation tools."""
    return tuple(get_model_adapter(name) for name in sorted(MODEL_REGISTRY))


def get_model(
    name: str,
    num_classes: int,
    pretrained: bool = False,
    *,
    task: str = "classification",
) -> nn.Module:
    """Create a model whose output is raw logits sized for the selected task."""
    if not isinstance(name, str):
        raise ValueError("Model name must be a string.")
    if isinstance(num_classes, bool) or not isinstance(num_classes, int) or num_classes < 1:
        raise ValueError("num_classes must be a positive integer.")
    if not isinstance(pretrained, bool):
        raise ValueError("pretrained must be a boolean.")
    normalized_task = normalize_task(task)
    adapter = get_model_adapter(name)
    if normalized_task not in adapter.supported_tasks:
        supported = ", ".join(adapter.supported_tasks)
        raise ValueError(
            f"Model {adapter.identifier!r} does not support task {normalized_task!r}; "
            f"supported tasks: {supported}."
        )
    if pretrained and not adapter.supports_pretrained:
        raise ValueError(f"Model {adapter.identifier!r} does not support pretrained weights.")

    cache_path = adapter.pretrained_cache_path
    offline = os.environ.get("UATF_OFFLINE", "").strip().lower() in {"1", "true", "yes"}
    if pretrained and offline and cache_path is not None and not cache_path.is_file():
        raise RuntimeError(
            f"Pretrained weights for model {adapter.identifier!r} are not cached. "
            f"Expected cache file: {cache_path}. Offline mode is enabled by UATF_OFFLINE. "
            "No random-weight fallback was used."
        )

    builder = MODEL_REGISTRY[adapter.identifier]
    try:
        model = builder(num_classes, pretrained)
        if not isinstance(model, nn.Module):
            raise TypeError(
                f"Model builder {adapter.identifier!r} must return a torch.nn.Module."
            )
        return model
    except Exception as error:
        if not pretrained:
            raise
        source = adapter.pretrained_source or "the registered builder's pretrained weights"
        cache_text = f" Expected cache file: {cache_path}." if cache_path else ""
        raise RuntimeError(
            f"Could not load pretrained weights for model {adapter.identifier!r} from {source}."
            f"{cache_text} Check the network or local cache. No random-weight fallback was used."
        ) from error


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
