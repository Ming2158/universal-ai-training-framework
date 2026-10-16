"""Small, reusable image preprocessing pipelines."""

from __future__ import annotations

from collections.abc import Sequence
from math import isfinite
from typing import Any

from torchvision import transforms
from torchvision.transforms import InterpolationMode

DEFAULT_AUGMENTATION = {
    "enabled": False,
    "horizontal_flip_probability": 0.5,
    "rotation_degrees": 0.0,
    "brightness": 0.0,
    "contrast": 0.0,
    "saturation": 0.0,
    "hue": 0.0,
}


def build_image_transform(
    image_size: int | tuple[int, int] | None = (224, 224),
    mean: Sequence[float] = (0.485, 0.456, 0.406),
    std: Sequence[float] = (0.229, 0.224, 0.225),
    *,
    augmentation: dict[str, Any] | None = None,
    training: bool = False,
    resize_enabled: bool = True,
) -> transforms.Compose:
    """Build image preprocessing; resizing and random augmentation are configurable."""
    if not isinstance(resize_enabled, bool):
        raise ValueError("resize_enabled must be a boolean.")
    size = (image_size, image_size) if isinstance(image_size, int) else image_size
    if resize_enabled and (
        size is None
        or len(size) != 2
        or any(isinstance(dimension, bool) or dimension < 1 for dimension in size)
    ):
        raise ValueError("image_size must contain two positive dimensions when resizing is enabled.")
    if len(mean) != 3 or len(std) != 3 or any(value <= 0 for value in std):
        raise ValueError("mean and std must each contain three RGB values; std must be positive.")

    operations = []
    options = {**DEFAULT_AUGMENTATION, **(augmentation or {})}
    if training and options["enabled"]:
        flip_probability = _finite_number(
            "horizontal_flip_probability", options["horizontal_flip_probability"]
        )
        rotation_degrees = _finite_number("rotation_degrees", options["rotation_degrees"])
        if not 0 <= flip_probability <= 1:
            raise ValueError("horizontal_flip_probability must be between 0 and 1.")
        if not 0 <= rotation_degrees <= 180:
            raise ValueError("rotation_degrees must be between 0 and 180.")
        if flip_probability:
            operations.append(transforms.RandomHorizontalFlip(p=flip_probability))
        if rotation_degrees:
            operations.append(
                transforms.RandomRotation(
                    degrees=rotation_degrees,
                    interpolation=InterpolationMode.BILINEAR,
                )
            )

        color_values = {
            key: _finite_number(key, options[key])
            for key in ("brightness", "contrast", "saturation", "hue")
        }
        if any(value < 0 for value in color_values.values()) or color_values["hue"] > 0.5:
            raise ValueError("Color jitter values must be non-negative and hue cannot exceed 0.5.")
        if any(color_values.values()):
            operations.append(
                transforms.ColorJitter(
                    brightness=color_values["brightness"] or None,
                    contrast=color_values["contrast"] or None,
                    saturation=color_values["saturation"] or None,
                    hue=color_values["hue"],
                )
            )

    if resize_enabled:
        operations.append(transforms.Resize(tuple(size)))
    operations.extend(
        [transforms.ToTensor(), transforms.Normalize(mean=tuple(mean), std=tuple(std))]
    )
    return transforms.Compose(operations)


def _finite_number(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value):
        raise ValueError(f"{name} must be a finite number.")
    return float(value)
