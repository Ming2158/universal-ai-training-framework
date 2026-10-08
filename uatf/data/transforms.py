"""Small, reusable image preprocessing pipelines."""

from __future__ import annotations

from collections.abc import Sequence

from torchvision import transforms


def build_image_transform(
    image_size: int | tuple[int, int] = (224, 224),
    mean: Sequence[float] = (0.485, 0.456, 0.406),
    std: Sequence[float] = (0.229, 0.224, 0.225),
) -> transforms.Compose:
    """Resize RGB images, convert them to tensors, and normalize each channel."""
    size = (image_size, image_size) if isinstance(image_size, int) else image_size
    if len(size) != 2 or any(dimension < 1 for dimension in size):
        raise ValueError("image_size must contain two positive dimensions.")
    if len(mean) != 3 or len(std) != 3 or any(value <= 0 for value in std):
        raise ValueError("mean and std must each contain three RGB values; std must be positive.")

    return transforms.Compose(
        [
            transforms.Resize(tuple(size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=tuple(mean), std=tuple(std)),
        ]
    )
