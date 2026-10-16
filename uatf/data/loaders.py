"""Deterministic dataset splitting and DataLoader construction."""

from __future__ import annotations

from math import floor
from typing import Callable

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, Subset, default_collate, random_split


def pad_image_batch(images: list[Tensor]) -> Tensor:
    """Pad variable-sized normalized images to the largest shape in the batch."""
    if not images:
        raise ValueError("An image batch cannot be empty.")
    channels = images[0].shape[0]
    if any(image.ndim != 3 or image.shape[0] != channels for image in images):
        raise ValueError("Images in a batch must have the same channel count and three dimensions.")
    height = max(image.shape[-2] for image in images)
    width = max(image.shape[-1] for image in images)
    height = max(32, (height + 31) // 32 * 32)
    width = max(32, (width + 31) // 32 * 32)
    batch = images[0].new_zeros((len(images), channels, height, width))
    for index, image in enumerate(images):
        batch[index, :, : image.shape[-2], : image.shape[-1]] = image
    return batch


def collate_image_batch(samples: list[tuple[Tensor, Tensor]]) -> tuple[Tensor, Tensor]:
    """Pad labeled image samples while preserving their batched targets."""
    images, targets = zip(*samples)
    return pad_image_batch(list(images)), default_collate(list(targets))


def collate_unlabeled_image_batch(images: list[Tensor]) -> Tensor:
    """Pad image-only samples used by directory inference."""
    return pad_image_batch(images)


def _loader_options(num_workers: int, pin_memory: bool) -> dict[str, object]:
    options: dict[str, object] = {"num_workers": num_workers, "pin_memory": pin_memory}
    if num_workers > 0:
        options.update(persistent_workers=True, prefetch_factor=2)
    return options


def split_dataset(
    dataset: Dataset,
    validation_fraction: float = 0.1,
    test_fraction: float = 0.0,
    seed: int = 42,
) -> tuple[Subset, Subset | None, Subset | None]:
    """Split a dataset into disjoint train, validation, and optional test subsets."""
    if not 0 <= validation_fraction < 1 or not 0 <= test_fraction < 1:
        raise ValueError("Validation and test fractions must be between 0 and 1.")
    if validation_fraction + test_fraction >= 1:
        raise ValueError("Validation and test fractions must sum to less than 1.")

    count = len(dataset)
    if count == 0:
        raise ValueError("Cannot split an empty dataset.")

    validation_count = floor(count * validation_fraction)
    test_count = floor(count * test_fraction)
    if validation_fraction > 0:
        validation_count = max(1, validation_count)
    if test_fraction > 0:
        test_count = max(1, test_count)
    training_count = count - validation_count - test_count
    if training_count < 1:
        raise ValueError(
            f"Dataset has {count} samples, too few for the requested train/validation/test split."
        )

    lengths = [training_count]
    if validation_fraction > 0:
        lengths.append(validation_count)
    if test_fraction > 0:
        lengths.append(test_count)
    subsets = random_split(
        dataset,
        lengths,
        generator=torch.Generator().manual_seed(seed),
    )
    training_subset = subsets[0]
    cursor = 1
    validation_subset = None
    test_subset = None
    if validation_fraction > 0:
        validation_subset = subsets[cursor]
        cursor += 1
    if test_fraction > 0:
        test_subset = subsets[cursor]
    return training_subset, validation_subset, test_subset


def create_data_loaders(
    dataset: Dataset,
    batch_size: int = 32,
    validation_fraction: float = 0.1,
    test_fraction: float = 0.0,
    seed: int = 42,
    num_workers: int = 0,
    train_transform: Callable | None = None,
    evaluation_transform: Callable | None = None,
    *,
    pad_variable_images: bool = False,
    pin_memory: bool = False,
) -> tuple[DataLoader, DataLoader | None, DataLoader | None]:
    """Create shuffled training and ordered evaluation loaders from any Dataset."""
    if batch_size < 1:
        raise ValueError("batch_size must be greater than 0.")
    if num_workers < 0:
        raise ValueError("num_workers cannot be negative.")

    train_set, validation_set, test_set = split_dataset(
        dataset,
        validation_fraction=validation_fraction,
        test_fraction=test_fraction,
        seed=seed,
    )
    if train_transform is not None or evaluation_transform is not None:
        with_transform = getattr(dataset, "with_transform", None)
        if not callable(with_transform):
            raise ValueError(
                "Separate training and evaluation transforms require a dataset with "
                "a with_transform() method."
            )
        if train_transform is not None:
            train_set = Subset(with_transform_dataset(dataset, train_transform), train_set.indices)
        if evaluation_transform is not None:
            evaluation_dataset = with_transform_dataset(dataset, evaluation_transform)
            if validation_set is not None:
                validation_set = Subset(evaluation_dataset, validation_set.indices)
            if test_set is not None:
                test_set = Subset(evaluation_dataset, test_set.indices)
    loader_options = _loader_options(num_workers, pin_memory)
    collate_fn = collate_image_batch if pad_variable_images else None
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
        collate_fn=collate_fn,
        **loader_options,
    )
    validation_loader = (
        DataLoader(
            validation_set,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            **loader_options,
        )
        if validation_set is not None
        else None
    )
    test_loader = (
        DataLoader(
            test_set,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            **loader_options,
        )
        if test_set is not None
        else None
    )
    return train_loader, validation_loader, test_loader


def with_transform_dataset(dataset: Dataset, transform: Callable) -> Dataset:
    """Return an image dataset copy that uses one independent transform pipeline."""
    method = getattr(dataset, "with_transform", None)
    if not callable(method):
        raise ValueError(
            "Separate training and evaluation transforms require a dataset with "
            "a with_transform() method."
        )
    return method(transform)
