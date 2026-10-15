from __future__ import annotations

import csv
from pathlib import Path

from PIL import Image

from uatf.data import (
    CsvImageClassificationDataset,
    CsvMultiLabelDataset,
    ImageFolderDataset,
    inspect_dataset,
    split_dataset,
)


def _save_image(path: Path, color: tuple[int, int, int] = (40, 80, 120)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (12, 8), color).save(path)


def _write_csv(path: Path, header: list[str], rows: list[list[str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(header)
        writer.writerows(rows)


def test_folder_inspection_matches_training_split_and_distribution(tmp_path: Path) -> None:
    for class_name in ("cats", "dogs"):
        for index in range(5):
            _save_image(tmp_path / class_name / f"{index}.png")

    inspection = inspect_dataset(tmp_path, "image_folder", validation_fraction=0.2, seed=29)
    dataset = ImageFolderDataset(tmp_path)
    expected_train, expected_validation, _ = split_dataset(
        dataset, validation_fraction=0.2, seed=29
    )

    assert inspection.can_train
    assert inspection.class_names == ["cats", "dogs"]
    assert inspection.sample_count == 10
    assert inspection.valid_sample_count == 10
    assert inspection.train_count == len(expected_train) == 8
    assert inspection.validation_count == len(expected_validation) == 2
    assert inspection.train_indices == expected_train.indices
    assert inspection.validation_indices == expected_validation.indices
    assert [row["count"] for row in inspection.class_counts] == [5, 5]
    assert all(row["ratio"] == 0.5 for row in inspection.class_counts)


def test_csv_inspection_supports_single_and_multi_label_formats(tmp_path: Path) -> None:
    rows = []
    for index in range(8):
        filename = f"{index}.png"
        _save_image(tmp_path / "images" / filename)
        rows.append([filename, "cat" if index < 4 else "dog"])
    single_csv = tmp_path / "single.csv"
    _write_csv(single_csv, ["image_path", "label"], rows)

    single = inspect_dataset(
        single_csv,
        "csv_classification",
        validation_fraction=0.25,
        seed=7,
        image_root=tmp_path / "images",
    )
    single_dataset = CsvImageClassificationDataset(
        single_csv, image_root=tmp_path / "images"
    )
    expected_train, expected_validation, _ = split_dataset(
        single_dataset, validation_fraction=0.25, seed=7
    )
    assert single.can_train
    assert single.class_names == ["cat", "dog"]
    assert single.train_indices == expected_train.indices
    assert single.validation_indices == expected_validation.indices

    multi_rows = [
        [f"{index}.png", str(index % 2), str((index // 2) % 2)]
        for index in range(8)
    ]
    multi_csv = tmp_path / "multi.csv"
    _write_csv(multi_csv, ["image_path", "cat", "dog"], multi_rows)
    multi = inspect_dataset(
        multi_csv,
        "csv_multilabel",
        validation_fraction=0.25,
        seed=7,
        image_root=tmp_path / "images",
    )
    multi_dataset = CsvMultiLabelDataset(multi_csv, image_root=tmp_path / "images")
    expected_train, expected_validation, _ = split_dataset(
        multi_dataset, validation_fraction=0.25, seed=7
    )
    assert multi.can_train
    assert multi.class_names == ["cat", "dog"]
    assert multi.train_indices == expected_train.indices
    assert multi.validation_indices == expected_validation.indices
    assert [row["count"] for row in multi.class_counts] == [4, 4]


def test_csv_report_collects_multiple_blocking_sample_errors(tmp_path: Path) -> None:
    _save_image(tmp_path / "ok.png")
    _save_image(tmp_path / "bad-label.png")
    (tmp_path / "broken.png").write_bytes(b"not an image")
    csv_path = tmp_path / "annotations.csv"
    _write_csv(
        csv_path,
        ["image_path", "cat", "dog"],
        [
            ["ok.png", "1", "0"],
            ["missing.png", "0", "1"],
            ["broken.png", "1", "0"],
            ["bad-label.png", "2", "0"],
            ["ok.png", "0", "1"],
        ],
    )

    inspection = inspect_dataset(csv_path, "csv_multilabel")
    report = inspection.to_dict()

    assert not inspection.can_train
    assert inspection.sample_count == 5
    assert inspection.valid_sample_count == 1
    assert report["status"] == "blocked"
    assert report["invalid_labels"]
    assert report["missing_files"] == [str((tmp_path / "missing.png").resolve())]
    assert report["unreadable_images"] == [str((tmp_path / "broken.png").resolve())]
    assert report["duplicate_paths"] == [str((tmp_path / "ok.png").resolve())]


def test_fixed_train_validation_directories_are_not_randomly_resplit(tmp_path: Path) -> None:
    for split_name in ("train", "val"):
        for class_name in ("cats", "dogs"):
            _save_image(tmp_path / split_name / class_name / "sample.png")

    inspection = inspect_dataset(
        tmp_path, "image_folder", validation_fraction=0.4, seed=11
    )

    assert inspection.can_train
    assert inspection.split_strategy == "fixed train/validation directories"
    assert inspection.train_count == 2
    assert inspection.validation_count == 2
    assert inspection.train_indices == [0, 1]
    assert inspection.validation_indices == [2, 3]


def test_nested_test_directories_warn_that_images_are_randomly_split(tmp_path: Path) -> None:
    for class_name in ("cats", "dogs"):
        for split_name, count in (("train", 3), ("test", 2)):
            for index in range(count):
                _save_image(tmp_path / class_name / split_name / f"{index}.png")

    inspection = inspect_dataset(tmp_path, "image_folder", validation_fraction=0.2, seed=29)

    assert inspection.can_train
    assert inspection.split_strategy == "random_split"
    assert inspection.sample_count == 10
    assert any(
        "'cats/test'" in warning
        and "'dogs/test'" in warning
        and "included in the random split" in warning
        for warning in inspection.warnings
    )

