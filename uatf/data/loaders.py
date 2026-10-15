"""Deterministic dataset splitting and DataLoader construction."""

from __future__ import annotations

from math import floor
from typing import Callable

import torch
from torch.utils.data import DataLoader, Dataset, Subset, random_split


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
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        generator=torch.Generator().manual_seed(seed),
    )
    validation_loader = (
        DataLoader(validation_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        if validation_set is not None
        else None
    )
    test_loader = (
        DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
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
