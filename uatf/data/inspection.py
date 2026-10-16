"""Dataset inspection shared by the local UI and programmatic workflows."""

from __future__ import annotations

import os
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

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

SPLIT_DIRECTORY_NAMES = {"train", "training", "val", "valid", "validation", "test", "testing"}


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
    image_content_verified: bool = True
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
            "image_content_verified": self.image_content_verified,
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


def _image_problem(
    report: DatasetInspection,
    path: Path,
    context: str,
    *,
    verify_image_content: bool,
) -> bool:
    try:
        _validate_image(path, context, verify_images=verify_image_content)
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
    *,
    verify_image_content: bool,
) -> bool:
    key = os.path.normcase(str(path))
    if key in seen:
        _append_once(report.duplicate_paths, str(path))
        _append_once(report.errors, f"{context}: duplicate image path: {path}")
        return False
    seen.add(key)
    if _image_problem(
        report, path, context, verify_image_content=verify_image_content
    ):
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


def _folder_parts(
    root: Path,
    report: DatasetInspection,
    progress_callback: Callable[[float | None, str], None] | None = None,
    nested_split_matches: set[str] | None = None,
) -> tuple[list[str], list[tuple[Path, str]]]:
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
    scanned_items = 0
    for directory in directories:
        image_paths: list[Path] = []
        unsupported_count = 0
        unsupported_examples: list[Path] = []
        for item in directory.rglob("*"):
            scanned_items += 1
            if progress_callback is not None and scanned_items % 500 == 0:
                progress_callback(
                    None,
                    f"扫描图片目录 | 已扫描 {scanned_items:,} 项，发现 {len(candidates) + len(image_paths):,} 张图片",
                )
            if item.is_dir():
                if nested_split_matches is not None and item.name.casefold() in SPLIT_DIRECTORY_NAMES:
                    nested_split_matches.add(
                        f"{directory.name}/{item.relative_to(directory).as_posix()}"
                    )
                continue
            if not item.is_file():
                continue
            if item.suffix.lower() in IMAGE_EXTENSIONS:
                image_paths.append(item.resolve())
            else:
                unsupported_count += 1
                if len(unsupported_examples) < 10:
                    unsupported_examples.append(item)
        image_paths.sort(key=str)
        if not image_paths:
            _append_once(report.abnormal_samples, f"Class directory contains no images: {directory}")
            _append_once(report.warnings, f"Class directory contains no images: {directory}")
        candidates.extend((image_path, directory.name) for image_path in image_paths)
        for item in unsupported_examples:
            _append_once(report.abnormal_samples, f"Unsupported file ignored: {item}")
        if unsupported_count:
            _append_once(
                report.warnings,
                f"{unsupported_count} unsupported file(s) in class directory {directory.name!r} were ignored.",
            )
    root_files = [item for item in root.iterdir() if item.is_file()]
    for item in root_files:
        _append_once(report.abnormal_samples, f"File outside class directory ignored: {item}")
    if root_files:
        _append_once(report.warnings, f"{len(root_files)} file(s) outside class directories were ignored.")
    return class_names, candidates


def _inspect_image_folder(
    report: DatasetInspection,
    progress_callback: Callable[[float | None, str], None] | None = None,
    *,
    verify_image_content: bool,
) -> None:
    root = _folder_root(report.path)
    fixed_validation = next(
        (root / name for name in ("validation", "val") if (root / name).is_dir()), None
    )
    if fixed_validation is not None and (root / "train").is_dir():
        train_root = root / "train"
        train_classes, train_candidates = _folder_parts(train_root, report, progress_callback)
        validation_report = DatasetInspection(path=fixed_validation, format="image_folder")
        validation_classes, validation_candidates = _folder_parts(
            fixed_validation, validation_report, progress_callback
        )
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
        total_candidates = len(train_candidates) + len(validation_candidates)
        if progress_callback is not None:
            description = (
                "开始验证图片文件内容"
                if verify_image_content
                else "读取图片清单，跳过内容验证"
            )
            progress_callback(0.2, f"已发现 {total_candidates:,} 张图片，{description}")
        checked = 0
        for path, name in train_candidates:
            checked += 1
            if progress_callback is not None and (
                checked == 1 or checked % max(1, total_candidates // 100) == 0
            ):
                progress_callback(
                    0.2 + 0.7 * checked / max(total_candidates, 1),
                    f"{'验证图片' if verify_image_content else '读取图片清单'} {checked:,}/{total_candidates:,}",
                )
            if name in class_to_idx:
                _add_sample(
                    report,
                    path,
                    class_to_idx[name],
                    seen,
                    "Training folder",
                    verify_image_content=verify_image_content,
                )
        report.train_count = len(report.samples)
        train_samples = len(report.samples)
        validation_seen: set[str] = set()
        for path, name in validation_candidates:
            checked += 1
            if progress_callback is not None and (
                checked == 1 or checked % max(1, total_candidates // 100) == 0 or checked == total_candidates
            ):
                progress_callback(
                    0.2 + 0.7 * checked / max(total_candidates, 1),
                    f"{'验证图片' if verify_image_content else '读取图片清单'} {checked:,}/{total_candidates:,}",
                )
            if name in class_to_idx:
                _add_sample(
                    report,
                    path,
                    class_to_idx[name],
                    validation_seen,
                    "Validation folder",
                    verify_image_content=verify_image_content,
                )
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
        nested_split_matches: set[str] = set()
        report.class_names, candidates = _folder_parts(
            root,
            report,
            progress_callback,
            nested_split_matches=nested_split_matches,
        )
        nested_split_directories = sorted(nested_split_matches)
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
        report.sample_count = len(candidates)
        class_to_idx = {name: index for index, name in enumerate(report.class_names)}
        seen: set[str] = set()
        if progress_callback is not None:
            description = (
                "开始验证图片文件内容"
                if verify_image_content
                else "读取图片清单，跳过内容验证"
            )
            progress_callback(0.2, f"已发现 {len(candidates):,} 张图片，{description}")
        for sample_index, (path, name) in enumerate(candidates, start=1):
            if progress_callback is not None and (
                sample_index == 1
                or sample_index % max(1, len(candidates) // 100) == 0
                or sample_index == len(candidates)
            ):
                progress_callback(
                    0.2 + 0.7 * sample_index / max(len(candidates), 1),
                    f"{'验证图片' if verify_image_content else '读取图片清单'} {sample_index:,}/{len(candidates):,}",
                )
            _add_sample(
                report,
                path,
                class_to_idx[name],
                seen,
                "Image-folder data",
                verify_image_content=verify_image_content,
            )


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
    progress_callback: Callable[[float | None, str], None] | None = None,
    verify_image_content: bool,
) -> None:
    try:
        if progress_callback is not None:
            progress_callback(0.01, "读取 CSV 标注")
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
    progress_interval = max(1, len(rows) // 100)
    for sample_index, (line_number, row) in enumerate(rows, start=1):
        if progress_callback is not None and (
            sample_index == 1
            or sample_index % progress_interval == 0
            or sample_index == len(rows)
        ):
            fraction = 0.02 + 0.88 * sample_index / max(len(rows), 1)
            progress_callback(
                fraction,
                f"{'验证图片' if verify_image_content else '读取标注'} {sample_index:,}/{len(rows):,}",
            )
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
        image_is_bad = _image_problem(
            report,
            path,
            f"CSV row {line_number}",
            verify_image_content=verify_image_content,
        )

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

    if progress_callback is not None:
        progress_callback(0.94, "汇总标签与训练/验证划分")

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
    progress_callback: Callable[[float | None, str], None] | None = None,
    verify_image_content: bool = True,
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
        image_content_verified=verify_image_content,
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
        if progress_callback is not None:
            progress_callback(0.01, "扫描图片目录")
        _inspect_image_folder(
            report,
            progress_callback,
            verify_image_content=verify_image_content,
        )
    else:
        _inspect_csv(
            report,
            image_root=image_root,
            label_columns=label_columns,
            progress_callback=progress_callback,
            verify_image_content=verify_image_content,
        )
    if not report.samples and not report.errors:
        _append_once(report.errors, "Dataset contains no valid image samples.")
    if progress_callback is not None:
        progress_callback(0.97, "生成数据集统计")
    result = _finish_report(report)
    if progress_callback is not None:
        progress_callback(1.0, "数据集检查完成")
    return result
