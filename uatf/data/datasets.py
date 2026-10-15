"""Reusable image datasets and input validation for classification tasks."""

from __future__ import annotations

import copy
import csv
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Callable, TypeAlias

import torch
from PIL import Image, UnidentifiedImageError
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import transforms

ImageTarget: TypeAlias = int | tuple[int, ...]
ImageSample: TypeAlias = tuple[Path, ImageTarget]

IMAGE_EXTENSIONS = {".bmp", ".gif", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


class DatasetValidationError(ValueError):
    """Raised when a dataset's files or labels do not match its format."""


def _read_csv(csv_file: Path) -> tuple[list[str], list[tuple[int, list[str]]]]:
    if not csv_file.is_file():
        raise DatasetValidationError(f"Dataset CSV does not exist: {csv_file}")

    try:
        with csv_file.open("r", encoding="utf-8-sig", newline="") as source:
            reader = csv.reader(source)
            try:
                header = [column.strip() for column in next(reader)]
            except StopIteration as error:
                raise DatasetValidationError(f"Dataset CSV is empty: {csv_file}") from error
            rows = [
                (line_number, [value.strip() for value in row])
                for line_number, row in enumerate(reader, start=2)
                if any(value.strip() for value in row)
            ]
    except (OSError, UnicodeError, csv.Error) as error:
        raise DatasetValidationError(f"Could not read dataset CSV {csv_file}: {error}") from error

    if not header or any(not column for column in header):
        raise DatasetValidationError("Dataset CSV contains an empty column name.")
    if len(set(header)) != len(header):
        raise DatasetValidationError("Dataset CSV contains duplicate column names.")
    return header, rows


def _resolve_image_path(value: str, image_root: Path, row_number: int) -> Path:
    if not value:
        raise DatasetValidationError(f"CSV row {row_number} has an empty image_path.")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = image_root / path
    return path.resolve()


def _validate_image(path: Path, context: str, verify_images: bool) -> None:
    if not path.is_file():
        raise DatasetValidationError(f"{context}: image file does not exist: {path}")
    if not verify_images:
        return

    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            image.load()
    except (OSError, SyntaxError, ValueError, UnidentifiedImageError) as error:
        raise DatasetValidationError(f"{context}: image is damaged or unreadable: {path}") from error


class _ImageDataset(Dataset[tuple[Tensor, Tensor]]):
    def __init__(
        self,
        samples: list[ImageSample],
        transform: Callable | None,
        *,
        verify_images: bool,
    ) -> None:
        seen_paths: set[str] = set()
        for path, _ in samples:
            key = os.path.normcase(str(path))
            if key in seen_paths:
                raise DatasetValidationError(f"Dataset contains a duplicate image path: {path}")
            seen_paths.add(key)
            _validate_image(path, "Dataset image", verify_images)

        if not samples:
            raise DatasetValidationError("Dataset contains no images.")

        self.samples = samples
        self.transform = transform or transforms.ToTensor()

    def with_transform(self, transform: Callable) -> _ImageDataset:
        """Return a dataset view sharing validated samples with its own transform."""
        if not callable(transform):
            raise TypeError("Dataset transform must be callable.")
        view = copy.copy(self)
        view.transform = transform
        return view

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        path, target = self.samples[index]
        try:
            with Image.open(path) as source:
                image = source.convert("RGB")
        except (OSError, SyntaxError, ValueError, UnidentifiedImageError) as error:
            raise DatasetValidationError(f"Could not read image: {path}") from error

        image_tensor = self.transform(image)
        if isinstance(target, tuple):
            target_tensor = torch.tensor(target, dtype=torch.float32)
        else:
            target_tensor = torch.tensor(target, dtype=torch.long)
        return image_tensor, target_tensor


class ImageFolderDataset(_ImageDataset):
    """Read single-label images from one class subdirectory per label."""

    def __init__(
        self,
        root: str | Path,
        transform: Callable | None = None,
        *,
        verify_images: bool = True,
    ) -> None:
        root_path = Path(root).expanduser().resolve()
        if not root_path.is_dir():
            raise DatasetValidationError(f"Image dataset directory does not exist: {root_path}")
        self.root = root_path

        class_directories = sorted(
            (path for path in root_path.iterdir() if path.is_dir()),
            key=lambda path: path.name,
        )
        if not class_directories:
            raise DatasetValidationError(
                f"Image dataset must contain one subdirectory per class: {root_path}"
            )

        self.classes = [directory.name for directory in class_directories]
        self.class_to_idx = {name: index for index, name in enumerate(self.classes)}
        samples: list[ImageSample] = []
        for directory in class_directories:
            image_paths = sorted(
                (
                    path.resolve()
                    for path in directory.rglob("*")
                    if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
                ),
                key=str,
            )
            samples.extend(
                (path, self.class_to_idx[directory.name]) for path in image_paths
            )

        super().__init__(samples, transform, verify_images=verify_images)


class CsvImageClassificationDataset(_ImageDataset):
    """Read single-label images from an ``image_path,label`` CSV file."""

    def __init__(
        self,
        csv_file: str | Path,
        image_root: str | Path | None = None,
        transform: Callable | None = None,
        *,
        class_names: Sequence[str] | None = None,
        verify_images: bool = True,
    ) -> None:
        csv_path = Path(csv_file).expanduser().resolve()
        header, rows = _read_csv(csv_path)
        required = {"image_path", "label"}
        missing = required.difference(header)
        if missing:
            raise DatasetValidationError(
                f"Dataset CSV is missing required column: {sorted(missing)[0]}"
            )
        if set(header) != required:
            unexpected = sorted(set(header).difference(required))
            raise DatasetValidationError(
                f"Single-label CSV has unexpected column: {unexpected[0]}"
            )

        root = Path(image_root).expanduser() if image_root is not None else csv_path.parent
        if not root.is_absolute():
            root = csv_path.parent / root
        root = root.resolve()
        self.image_root = root
        path_column = header.index("image_path")
        label_column = header.index("label")
        raw_rows: list[tuple[int, Path, str]] = []
        labels: set[str] = set()
        seen_paths: set[str] = set()
        for row_number, row in rows:
            if len(row) != len(header):
                raise DatasetValidationError(
                    f"CSV row {row_number} has {len(row)} fields; expected {len(header)}."
                )
            image_path = _resolve_image_path(row[path_column], root, row_number)
            label = row[label_column]
            if not label:
                raise DatasetValidationError(f"CSV row {row_number} has an empty label.")
            key = os.path.normcase(str(image_path))
            if key in seen_paths:
                raise DatasetValidationError(
                    f"CSV row {row_number} repeats image_path: {image_path}"
                )
            seen_paths.add(key)
            raw_rows.append((row_number, image_path, label))
            labels.add(label)

        if not raw_rows:
            raise DatasetValidationError("Dataset CSV contains no image rows.")
        self.classes = sorted(labels) if class_names is None else list(class_names)
        if (
            not self.classes
            or any(not isinstance(name, str) or not name for name in self.classes)
            or len(set(self.classes)) != len(self.classes)
        ):
            raise DatasetValidationError("class_names must contain unique, non-empty labels.")
        unknown_labels = labels.difference(self.classes)
        if unknown_labels:
            raise DatasetValidationError(
                f"CSV contains label not present in the model asset: {sorted(unknown_labels)[0]}"
            )
        self.class_to_idx = {name: index for index, name in enumerate(self.classes)}
        samples = [
            (path, self.class_to_idx[label])
            for _, path, label in raw_rows
        ]
        for row_number, path, _ in raw_rows:
            _validate_image(path, f"CSV row {row_number}", verify_images)
        super().__init__(samples, transform, verify_images=False)


class CsvMultiLabelDataset(_ImageDataset):
    """Read multi-hot targets from ``image_path`` plus binary label columns."""

    def __init__(
        self,
        csv_file: str | Path,
        image_root: str | Path | None = None,
        transform: Callable | None = None,
        *,
        label_columns: list[str] | None = None,
        verify_images: bool = True,
    ) -> None:
        csv_path = Path(csv_file).expanduser().resolve()
        header, rows = _read_csv(csv_path)
        if "image_path" not in header:
            raise DatasetValidationError(
                "Dataset CSV is missing required column: image_path"
            )

        selected_labels = (
            label_columns
            if label_columns is not None
            else [column for column in header if column != "image_path"]
        )
        if not selected_labels:
            raise DatasetValidationError("Multi-label CSV must contain at least one label column.")
        if len(set(selected_labels)) != len(selected_labels):
            raise DatasetValidationError("Multi-label columns contain duplicate names.")
        missing = [column for column in selected_labels if column not in header]
        if missing:
            raise DatasetValidationError(
                f"Dataset CSV is missing label column: {missing[0]}"
            )
        unexpected = set(header).difference({"image_path", *selected_labels})
        if unexpected:
            raise DatasetValidationError(
                f"Multi-label CSV has unexpected column: {sorted(unexpected)[0]}"
            )

        root = Path(image_root).expanduser() if image_root is not None else csv_path.parent
        if not root.is_absolute():
            root = csv_path.parent / root
        root = root.resolve()
        self.image_root = root
        path_column = header.index("image_path")
        label_indices = [header.index(column) for column in selected_labels]
        samples: list[ImageSample] = []
        seen_paths: set[str] = set()
        for row_number, row in rows:
            if len(row) != len(header):
                raise DatasetValidationError(
                    f"CSV row {row_number} has {len(row)} fields; expected {len(header)}."
                )
            path = _resolve_image_path(row[path_column], root, row_number)
            key = os.path.normcase(str(path))
            if key in seen_paths:
                raise DatasetValidationError(
                    f"CSV row {row_number} repeats image_path: {path}"
                )
            seen_paths.add(key)

            values: list[int] = []
            for column, index in zip(selected_labels, label_indices):
                value = row[index]
                if value not in {"0", "1"}:
                    raise DatasetValidationError(
                        f"CSV row {row_number} has invalid value {value!r} for "
                        f"'{column}'; expected 0 or 1."
                    )
                values.append(int(value))
            _validate_image(path, f"CSV row {row_number}", verify_images)
            samples.append((path, tuple(values)))

        if not samples:
            raise DatasetValidationError("Dataset CSV contains no image rows.")
        self.classes = list(selected_labels)
        self.class_to_idx = {name: index for index, name in enumerate(self.classes)}
        super().__init__(samples, transform, verify_images=False)
