from __future__ import annotations

import io
import stat
import zipfile
from pathlib import Path

import pytest
from PIL import Image

from app import app


def _image_bytes() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (10, 8), (80, 120, 160)).save(buffer, format="PNG")
    return buffer.getvalue()


def _write_image_dataset(root: Path) -> None:
    for class_name in ("cats", "dogs"):
        class_dir = root / class_name
        class_dir.mkdir(parents=True, exist_ok=True)
        for index in range(3):
            Image.new("RGB", (10, 8), (index * 20, 80, 160)).save(
                class_dir / f"{index}.png"
            )


def test_zip_import_is_cached_and_nested_dataset_root_is_resolved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    upload_root = tmp_path / "uploads"
    monkeypatch.setattr(app, "UPLOAD_ROOT", upload_root)
    archive_path = tmp_path / "dataset.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        for class_name in ("cats", "dogs"):
            archive.writestr(f"bundle/{class_name}/one.png", _image_bytes())

    first = app._safe_extract_dataset(str(archive_path), "classification", "image_folder")
    second = app._safe_extract_dataset(str(archive_path), "classification", "image_folder")
    inspection = app.inspect_dataset(first, "image_folder", validation_fraction=0.2, seed=3)

    assert first == second
    assert len([item for item in upload_root.iterdir() if item.is_dir()]) == 1
    assert inspection.path == (first / "bundle").resolve()
    assert inspection.can_train


def test_zip_import_rejects_path_traversal_and_cleans_partial_tree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    upload_root = tmp_path / "uploads"
    monkeypatch.setattr(app, "UPLOAD_ROOT", upload_root)
    archive_path = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("../escaped.txt", "not allowed")

    with pytest.raises(ValueError, match="不安全的文件路径"):
        app._safe_extract_dataset(str(archive_path), "classification")

    assert not (tmp_path / "escaped.txt").exists()
    assert not list(upload_root.iterdir())


def test_zip_import_rejects_symbolic_link_entries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    upload_root = tmp_path / "uploads"
    monkeypatch.setattr(app, "UPLOAD_ROOT", upload_root)
    archive_path = tmp_path / "link.zip"
    link = zipfile.ZipInfo("linked-file")
    link.create_system = 3
    link.external_attr = (stat.S_IFLNK | 0o777) << 16
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr(link, "target")

    with pytest.raises(ValueError, match="符号链接"):
        app._safe_extract_dataset(str(archive_path), "classification")

    assert not list(upload_root.iterdir())


def test_gui_inspection_returns_real_preview_and_persistent_dataset_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset_path = tmp_path / "dataset"
    _write_image_dataset(dataset_path)
    monkeypatch.setattr(app, "DATASET_REGISTRY", tmp_path / "registry")
    monkeypatch.setattr(app, "DATASET_ROOT", tmp_path / "ui_datasets")

    result = app.inspect_dataset_ui(
        None,
        str(dataset_path),
        "classification",
        "image_folder",
        0.25,
        13,
        16,
    )
    (
        state,
        index,
        summary,
        report,
        class_rows,
        chart_path,
        original,
        processed,
        information,
        confirmation,
        training_config_visibility,
    ) = result

    assert state["report"]["can_train"]
    assert report["valid_sample_count"] == 6
    assert report["split"]["train_count"] == 5
    assert report["split"]["validation_count"] == 1
    assert sum(row[1] for row in class_rows) == 6
    assert all(row[2] == "50.0%" for row in class_rows)
    assert chart_path and Path(chart_path).is_file()
    assert original.size == (10, 8)
    assert processed.size == (16, 16)
    assert str(dataset_path) in information
    assert "检查通过" in summary
    assert confirmation["value"] is False
    assert training_config_visibility["visible"] is True
    assert list((tmp_path / "registry").glob("*.json"))


def test_training_is_blocked_until_a_dataset_has_been_inspected() -> None:
    result = next(
        app.start_training(
            None,
            None,
            None,
            "classification",
            "image_folder",
            0.2,
            42,
            224,
            False,
            "resnet50",
            1,
            2,
            0.001,
            "cpu",
        )
    )

    assert result[0].startswith("错误：请先导入并检查数据集")


def test_blocking_inspection_error_prevents_training(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset_path = tmp_path / "empty"
    dataset_path.mkdir()
    monkeypatch.setattr(app, "DATASET_REGISTRY", tmp_path / "registry")
    _, _, state = app._inspect_selected_dataset(
        None, str(dataset_path), "classification", "image_folder", 0.2, 42, 224
    )

    result = next(
        app.start_training(
            None,
            str(dataset_path),
            state,
            "classification",
            "image_folder",
            0.2,
            42,
            224,
            True,
            "resnet50",
            1,
            2,
            0.001,
            "cpu",
        )
    )

    assert "存在阻断性错误" in result[0]


def test_warning_confirmation_is_required_before_training(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset_path = tmp_path / "dataset"
    _write_image_dataset(dataset_path)
    (dataset_path / "empty-class").mkdir()
    monkeypatch.setattr(app, "DATASET_REGISTRY", tmp_path / "registry")
    _, report, state = app._inspect_selected_dataset(
        None, str(dataset_path), "classification", "image_folder", 0.25, 42, 224
    )
    assert report["warnings"]
    assert report["can_train"]

    result = next(
        app.start_training(
            None,
            str(dataset_path),
            state,
            "classification",
            "image_folder",
            0.25,
            42,
            224,
            False,
            "resnet50",
            1,
            2,
            0.001,
            "cpu",
        )
    )

    assert "阅读报告并勾选确认" in result[0]

