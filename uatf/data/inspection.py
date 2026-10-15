"""Dataset inspection shared by the local UI and programmatic workflows."""

from __future__ import annotations

import os
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from torch.utils.data import Dataset

from uatf.data.datasets import (
    IMAGE_EXTENSIONS,
    DatasetValidationError,
    ImageSample,
    _read_csv,
    _resolve_image_path,
    _validate_image,
)
from uatf.data.loaders import split_dataset


@dataclass
class DatasetInspection:
    """A validation report plus the valid, ordered samples used for preview/split."""

    path: Path
    format: str
    sample_count: int = 0
    samples: list[ImageSample] = field(default_factory=list)
    class_names: list[str] = field(default_factory=list)
    class_counts: list[dict[str, Any]] = field(default_factory=list)
    train_indices: list[int] = field(default_factory=list)
    validation_indices: list[int] = field(default_factory=list)
    train_count: int = 0
    validation_count: int = 0
    validation_fraction: float = 0.2
    seed: int = 42
    split_strategy: str = "random_split"
    annotation_csv: Path | None = None
    image_root: Path | None = None
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    invalid_labels: list[str] = field(default_factory=list)
    missing_files: list[str] = field(default_factory=list)
    unreadable_images: list[str] = field(default_factory=list)
    duplicate_paths: list[str] = field(default_factory=list)
    abnormal_samples: list[str] = field(default_factory=list)

    @property
    def valid_sample_count(self) -> int:
        return len(self.samples)

    @property
    def can_train(self) -> bool:
        return not self.errors and self.train_count > 0 and self.validation_count > 0

    def to_dict(self, *, include_samples: bool = False) -> dict[str, Any]:
        result = {
            "path": str(self.path),
            "format": self.format,
            "sample_count": self.sample_count,
            "valid_sample_count": self.valid_sample_count,
            "class_names": list(self.class_names),
            "class_counts": list(self.class_counts),
            "split": {
                "strategy": self.split_strategy,
                "train_count": self.train_count,
                "validation_count": self.validation_count,
                "validation_fraction": self.validation_fraction,
                "seed": self.seed,
            },
            "annotation_csv": str(self.annotation_csv) if self.annotation_csv else None,
            "image_root": str(self.image_root) if self.image_root else None,
            "status": "blocked" if self.errors else "warning" if self.warnings else "ready",
            "can_train": self.can_train,
            "errors": list(self.errors),
            "warnings": list(self.warnings),
            "invalid_labels": list(self.invalid_labels),
            "missing_files": list(self.missing_files),
            "unreadable_images": list(self.unreadable_images),
            "duplicate_paths": list(self.duplicate_paths),
            "abnormal_samples": list(self.abnormal_samples),
        }
        if include_samples:
            result["samples"] = []
            for path, target in self.samples:
                if isinstance(target, tuple):
                    label: str | list[str] = [
                        name
                        for name, enabled in zip(self.class_names, target, strict=True)
                        if enabled
                    ]
                else:
                    label = self.class_names[target]
                result["samples"].append({"path": str(path), "label": label})
        return result


class _ManifestDataset(Dataset):
    def __init__(self, samples: list[ImageSample]) -> None:
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> ImageSample:
        return self.samples[index]


def _append_once(values: list[str], value: str) -> None:
    if value not in values:
        values.append(value)


def _image_problem(report: DatasetInspection, path: Path, context: str) -> bool:
    try:
        _validate_image(path, context, verify_images=True)
    except DatasetValidationError as error:
        message = str(error)
        if "does not exist" in message:
            _append_once(report.missing_files, str(path))
        else:
            _append_once(report.unreadable_images, str(path))
        _append_once(report.errors, message)
        return True
    return False


def _add_sample(
    report: DatasetInspection,
    path: Path,
    target: int | tuple[int, ...],
    seen: set[str],
    context: str,
) -> bool:
    key = os.path.normcase(str(path))
    if key in seen:
        _append_once(report.duplicate_paths, str(path))
        _append_once(report.errors, f"{context}: duplicate image path: {path}")
        return False
    seen.add(key)
    if _image_problem(report, path, context):
        return False
    report.samples.append((path, target))
    return True


def _folder_root(path: Path) -> Path:
    if not path.is_dir():
        return path
    for validation_name in ("validation", "val"):
        if (path / "train").is_dir() and (path / validation_name).is_dir():
            return path
    if (path / "train").is_dir():
        return path / "train"
    child_directories = sorted(item for item in path.iterdir() if item.is_dir())
    if len(child_directories) == 1:
        nested_directories = [item for item in child_directories[0].iterdir() if item.is_dir()]
        if len(nested_directories) >= 2:
            return child_directories[0]
    return path


def _folder_parts(root: Path, report: DatasetInspection) -> tuple[list[str], list[tuple[Path, str]]]:
    if not root.is_dir():
        _append_once(report.errors, f"Image dataset directory does not exist: {root}")
        return [], []
    directories = sorted(
        (item for item in root.iterdir() if item.is_dir()), key=lambda item: item.name
    )
    if not directories:
        _append_once(
            report.errors,
            f"Image-folder data must contain one subdirectory per class: {root}",
        )
        return [], []

    class_names = [directory.name for directory in directories]
    candidates: list[tuple[Path, str]] = []
    for directory in directories:
        image_paths = sorted(
            (
                item.resolve()
                for item in directory.rglob("*")
                if item.is_file() and item.suffix.lower() in IMAGE_EXTENSIONS
            ),
            key=str,
        )
        if not image_paths:
            _append_once(report.abnormal_samples, f"Class directory contains no images: {directory}")
            _append_once(report.warnings, f"Class directory contains no images: {directory}")
        candidates.extend((image_path, directory.name) for image_path in image_paths)
        unsupported = [
            item
            for item in directory.rglob("*")
            if item.is_file() and item.suffix.lower() not in IMAGE_EXTENSIONS
        ]
        for item in unsupported:
            _append_once(report.abnormal_samples, f"Unsupported file ignored: {item}")
        if unsupported:
            _append_once(
                report.warnings,
                f"{len(unsupported)} unsupported file(s) in class directory {directory.name!r} were ignored.",
            )
    root_files = [item for item in root.iterdir() if item.is_file()]
    for item in root_files:
        _append_once(report.abnormal_samples, f"File outside class directory ignored: {item}")
    if root_files:
        _append_once(report.warnings, f"{len(root_files)} file(s) outside class directories were ignored.")
    return class_names, candidates


def _inspect_image_folder(report: DatasetInspection) -> None:
    root = _folder_root(report.path)
    fixed_validation = next(
        (root / name for name in ("validation", "val") if (root / name).is_dir()), None
    )
    if fixed_validation is not None and (root / "train").is_dir():
        train_root = root / "train"
        train_classes, train_candidates = _folder_parts(train_root, report)
        validation_report = DatasetInspection(path=fixed_validation, format="image_folder")
        validation_classes, validation_candidates = _folder_parts(fixed_validation, validation_report)
        if train_classes != validation_classes:
            _append_once(
                report.errors,
                "Training and validation folders must contain the same class directories in the same sorted order.",
            )
        for warning in validation_report.warnings:
            _append_once(report.warnings, warning)
        for abnormal in validation_report.abnormal_samples:
            _append_once(report.abnormal_samples, abnormal)
        report.class_names = train_classes
        report.sample_count = len(train_candidates) + len(validation_candidates)
        report.split_strategy = "fixed train/validation directories"
        report.path = root.resolve()
        report.image_root = train_root.resolve()
        class_to_idx = {name: index for index, name in enumerate(train_classes)}
        seen: set[str] = set()
        for path, name in train_candidates:
            if name in class_to_idx:
                _add_sample(report, path, class_to_idx[name], seen, "Training folder")
        report.train_count = len(report.samples)
        train_samples = len(report.samples)
        validation_seen: set[str] = set()
        for path, name in validation_candidates:
            if name in class_to_idx:
                _add_sample(report, path, class_to_idx[name], validation_seen, "Validation folder")
        overlap = seen.intersection(validation_seen)
        for key in sorted(overlap):
            path = next((str(item[0]) for item in report.samples if os.path.normcase(str(item[0])) == key), key)
            _append_once(report.duplicate_paths, path)
            _append_once(report.errors, f"Training and validation folders share the same image path: {path}")
        report.validation_count = len(report.samples) - train_samples
        report.train_indices = list(range(train_samples))
        report.validation_indices = list(range(train_samples, len(report.samples)))
    else:
        report.path = root.resolve()
        report.image_root = root.resolve()
        split_directory_names = {
            "train",
            "training",
            "val",
            "valid",
            "validation",
            "test",
            "testing",
        }
        nested_split_directories = sorted(
            {
                f"{class_directory.name}/{directory.relative_to(class_directory).as_posix()}"
                for class_directory in root.iterdir()
                if class_directory.is_dir()
                for directory in class_directory.rglob("*")
                if directory.is_dir() and directory.name.casefold() in split_directory_names
            }
        )
        if nested_split_directories:
            examples = ", ".join(repr(item) for item in nested_split_directories[:3])
            if len(nested_split_directories) > 3:
                examples += f", and {len(nested_split_directories) - 3} more"
            _append_once(
                report.warnings,
                "Nested split-like directories were found under class folders "
                f"({examples}). Their images are included in the random split; "
                "verify they are not a held-out test set.",
            )
        report.class_names, candidates = _folder_parts(root, report)
        report.sample_count = len(candidates)
        class_to_idx = {name: index for index, name in enumerate(report.class_names)}
        seen: set[str] = set()
        for path, name in candidates:
            _add_sample(report, path, class_to_idx[name], seen, "Image-folder data")


def _resolve_csv(path: Path, data_format: str) -> Path:
    if path.is_file():
        return path
    if not path.is_dir():
        raise DatasetValidationError(f"Dataset path does not exist: {path}")
    preferred = path / "annotations.csv"
    if preferred.is_file():
        return preferred
    candidates = sorted(path.glob("*.csv"))
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        raise DatasetValidationError(
            f"Dataset directory contains multiple CSV files; select the annotation CSV directly: {path}"
        )
    children = [item for item in path.iterdir() if item.is_dir()]
    if len(children) == 1:
        return _resolve_csv(children[0], data_format)
    name = (
        "annotations.csv"
        if data_format == "csv_multilabel"
        else "a CSV with image_path,label columns"
    )
    raise DatasetValidationError(f"CSV dataset is missing {name}: {path}")


def _inspect_csv(
    report: DatasetInspection,
    *,
    image_root: str | Path | None,
    label_columns: list[str] | None,
) -> None:
    try:
        csv_path = _resolve_csv(report.path, report.format)
        header, rows = _read_csv(csv_path)
    except (DatasetValidationError, OSError, ValueError) as error:
        _append_once(report.errors, str(error))
        return
    report.annotation_csv = csv_path.resolve()

    if report.format == "csv_classification":
        required = {"image_path", "label"}
        missing = required.difference(header)
        if missing:
            message = f"Dataset CSV is missing required column: {sorted(missing)[0]}"
            _append_once(report.errors, message)
            _append_once(report.abnormal_samples, message)
            return
        unexpected = sorted(set(header).difference(required))
        if unexpected:
            message = f"Single-label CSV has unexpected column: {unexpected[0]}"
            _append_once(report.errors, message)
            _append_once(report.abnormal_samples, message)
            return
        selected_labels = ["label"]
    else:
        if "image_path" not in header:
            message = "Dataset CSV is missing required column: image_path"
            _append_once(report.errors, message)
            _append_once(report.abnormal_samples, message)
            return
        selected_labels = label_columns if label_columns is not None else [col for col in header if col != "image_path"]
        if not selected_labels:
            _append_once(report.errors, "Multi-label CSV must contain at least one label column.")
            return
        if len(set(selected_labels)) != len(selected_labels):
            _append_once(report.errors, "Multi-label columns contain duplicate names.")
            return
        missing = [column for column in selected_labels if column not in header]
        if missing:
            _append_once(report.errors, f"Dataset CSV is missing label column: {missing[0]}")
            return
        unexpected = sorted(set(header).difference({"image_path", *selected_labels}))
        if unexpected:
            _append_once(report.errors, f"Multi-label CSV has unexpected column: {unexpected[0]}")
            return
    report.class_names = (
        [] if report.format == "csv_classification" else list(selected_labels)
    )

    if image_root is not None:
        resolved_image_root = Path(image_root).expanduser()
        if not resolved_image_root.is_absolute():
            resolved_image_root = csv_path.parent / resolved_image_root
        image_directory = resolved_image_root.resolve()
    else:
        configured_path = report.path
        if configured_path.is_dir() and (configured_path / "images").is_dir():
            image_directory = (configured_path / "images").resolve()
        else:
            candidate = csv_path.parent / "images"
            image_directory = candidate.resolve() if candidate.is_dir() else csv_path.parent.resolve()
    report.image_root = image_directory

    path_index = header.index("image_path")
    label_indices = [header.index(column) for column in selected_labels]
    report.sample_count = len(rows)
    raw_valid: list[tuple[Path, str | tuple[int, ...]]] = []
    seen: set[str] = set()
    for line_number, row in rows:
        if len(row) != len(header):
            message = f"CSV row {line_number} has {len(row)} fields; expected {len(header)}."
            _append_once(report.errors, message)
            _append_once(report.abnormal_samples, message)
            continue
        try:
            path = _resolve_image_path(row[path_index], image_directory, line_number)
        except DatasetValidationError as error:
            message = str(error)
            _append_once(report.errors, message)
            _append_once(report.abnormal_samples, message)
            continue
        key = os.path.normcase(str(path))
        if key in seen:
            _append_once(report.duplicate_paths, str(path))
            _append_once(report.errors, f"CSV row {line_number} repeats image_path: {path}")
            continue
        seen.add(key)
        image_is_bad = _image_problem(report, path, f"CSV row {line_number}")

        if report.format == "csv_classification":
            label = row[label_indices[0]]
            if not label:
                message = f"CSV row {line_number} has an empty label."
                _append_once(report.invalid_labels, message)
                _append_once(report.errors, message)
                continue
            target: str | tuple[int, ...] = label
        else:
            values: list[int] = []
            invalid = False
            for column, index in zip(selected_labels, label_indices, strict=True):
                value = row[index]
                if value not in {"0", "1"}:
                    message = (
                        f"CSV row {line_number} has invalid value {value!r} for '{column}'; "
                        "expected 0 or 1."
                    )
                    _append_once(report.invalid_labels, message)
                    _append_once(report.errors, message)
                    invalid = True
                else:
                    values.append(int(value))
            if invalid:
                continue
            target = tuple(values)
        if not image_is_bad:
            raw_valid.append((path, target))

    if report.format == "csv_classification":
        report.class_names = sorted({label for _, label in raw_valid if isinstance(label, str)})
        class_to_idx = {name: index for index, name in enumerate(report.class_names)}
        for path, label in raw_valid:
            assert isinstance(label, str)
            report.samples.append((path, class_to_idx[label]))
    else:
        for path, target in raw_valid:
            assert isinstance(target, tuple)
            report.samples.append((path, target))

    if not rows:
        _append_once(report.errors, "Dataset CSV contains no image rows.")
    if not report.samples:
        _append_once(report.errors, "Dataset contains no valid image samples.")


def _finish_report(report: DatasetInspection) -> DatasetInspection:
    report.sample_count = max(report.sample_count, len(report.samples))
    if not report.samples:
        return report
    manifest = _ManifestDataset(report.samples)
    if report.split_strategy == "fixed train/validation directories":
        train_samples = report.samples[: report.train_count]
        validation_samples = report.samples[report.train_count :]
        train_ids = {os.path.normcase(str(path)) for path, _ in train_samples}
        validation_ids = {os.path.normcase(str(path)) for path, _ in validation_samples}
        if train_ids.intersection(validation_ids):
            _append_once(report.errors, "Training and validation subsets overlap.")
        if not train_samples or not validation_samples:
            _append_once(report.errors, "Both training and validation folders must contain valid images.")
    else:
        try:
            train_set, validation_set, _ = split_dataset(
                manifest,
                validation_fraction=report.validation_fraction,
                seed=report.seed,
            )
        except ValueError as error:
            _append_once(report.errors, f"Cannot split dataset: {error}")
            return report
        assert validation_set is not None
        report.train_indices = list(train_set.indices)
        report.validation_indices = list(validation_set.indices)
        report.train_count = len(train_set)
        report.validation_count = len(validation_set)
        if set(report.train_indices).intersection(report.validation_indices):
            _append_once(report.errors, "Training and validation subsets overlap.")

    total = max(1, len(report.samples))
    if report.format == "csv_multilabel":
        counts = Counter(
            name
            for _, target in report.samples
            for name, enabled in zip(report.class_names, target, strict=True)
            if enabled
        )
        train_counts = Counter(
            name
            for index in report.train_indices
            for name, enabled in zip(report.class_names, report.samples[index][1], strict=True)
            if enabled
        )
        validation_counts = Counter(
            name
            for index in report.validation_indices
            for name, enabled in zip(report.class_names, report.samples[index][1], strict=True)
            if enabled
        )
    else:
        counts = Counter(report.class_names[target] for _, target in report.samples)
        train_counts = Counter(
            report.class_names[report.samples[index][1]] for index in report.train_indices
        )
        validation_counts = Counter(
            report.class_names[report.samples[index][1]] for index in report.validation_indices
        )
    report.class_counts = [
        {
            "label": label,
            "count": counts[label],
            "ratio": counts[label] / total,
            "train_count": train_counts[label],
            "validation_count": validation_counts[label],
        }
        for label in report.class_names
    ]
    missing_train = [label for label in report.class_names if train_counts[label] == 0]
    missing_validation = [label for label in report.class_names if validation_counts[label] == 0]
    if missing_train:
        _append_once(report.warnings, f"Some labels have no training samples: {', '.join(missing_train)}")
    if missing_validation:
        _append_once(report.warnings, f"Some labels have no validation samples: {', '.join(missing_validation)}")
    return report


def inspect_dataset(
    path: str | Path,
    data_format: str,
    *,
    validation_fraction: float = 0.2,
    seed: int = 42,
    image_root: str | Path | None = None,
    label_columns: list[str] | None = None,
) -> DatasetInspection:
    """Inspect an existing UATF image dataset without changing it or copying files."""
    normalized_format = data_format.strip().lower()
    supported = {"image_folder", "csv_classification", "csv_multilabel"}
    dataset_path = Path(path).expanduser().resolve()
    report = DatasetInspection(
        path=dataset_path,
        format=normalized_format,
        validation_fraction=validation_fraction,
        seed=seed,
    )
    if normalized_format not in supported:
        _append_once(report.errors, f"Unsupported dataset format: {data_format!r}.")
        return report
    if not 0 < validation_fraction < 1:
        _append_once(report.errors, "Validation fraction must be greater than 0 and less than 1.")
        return report
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        _append_once(report.errors, "Random seed must be a non-negative integer.")
        return report

    if normalized_format == "image_folder":
        _inspect_image_folder(report)
    else:
        _inspect_csv(report, image_root=image_root, label_columns=label_columns)
    if not report.samples and not report.errors:
        _append_once(report.errors, "Dataset contains no valid image samples.")
    return _finish_report(report)
