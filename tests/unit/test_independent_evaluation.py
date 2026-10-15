from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
import torch
import yaml
from PIL import Image
from torch import nn

from app import app
from uatf.evaluation import evaluate_model_asset
from uatf.experiments import (
    export_model_asset,
    list_model_assets,
    load_exported_model_asset,
    register_model_asset,
)
from uatf.inference import predict_model_asset_directory, predict_model_asset_image
from uatf.models.factory import MODEL_REGISTRY


def _builder(output_count: int, pretrained: bool) -> nn.Module:
    assert pretrained is False
    return nn.Sequential(nn.Flatten(), nn.Linear(3 * 4 * 4, output_count))


def _write_image(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (4, 4), color).save(path)


def _make_asset(
    root: Path,
    *,
    task: str = "classification",
    data_format: str = "image_folder",
    labels: list[str] | None = None,
) -> Path:
    label_names = labels or ["cat", "dog"]
    experiment = root / "experiment_001"
    experiment.mkdir(parents=True)
    configuration = {
        "task": {"type": task},
        "dataset": {
            "format": data_format,
            "image_size": 4,
            "mean": [0.5, 0.5, 0.5],
            "std": [0.25, 0.25, 0.25],
            "label_columns": label_names if task == "multi_label" else None,
        },
        "model": {"name": "phase6_tiny", "num_classes": len(label_names), "pretrained": True},
        "evaluation": {"threshold": 0.5},
    }
    config_path = experiment / "config.yaml"
    config_path.write_text(yaml.safe_dump(configuration), encoding="utf-8")
    model = _builder(len(label_names), False)
    with torch.no_grad():
        model[1].weight.zero_()
        model[1].bias.copy_(torch.tensor([2.0, -2.0]))
    checkpoint_path = experiment / "best_model.pth"
    torch.save({"model_state_dict": model.state_dict()}, checkpoint_path)
    (experiment / "metrics.json").write_text('{"training_result":"preserved"}', encoding="utf-8")
    (experiment / "experiment.json").write_text(
        json.dumps({"id": "phase6-run", "dataset": {"class_names": label_names}}),
        encoding="utf-8",
    )
    register_model_asset(
        root,
        {
            "id": "phase6-model",
            "name": "phase6-model",
            "architecture": "phase6_tiny",
            "experiment_id": "phase6-run",
            "experiment_directory": experiment.name,
            "weights_path": checkpoint_path,
            "configuration_path": config_path,
            "task": task,
            "class_names": label_names,
        },
    )
    return experiment


@pytest.fixture
def tiny_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(MODEL_REGISTRY, "phase6_tiny", _builder)


def test_independent_classification_evaluation_and_inference_are_isolated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, tiny_model: None
) -> None:
    root = tmp_path / "runs"
    experiment = _make_asset(root)
    dataset = tmp_path / "evaluation-data"
    _write_image(dataset / "cat" / "cat-1.png", (200, 40, 40))
    _write_image(dataset / "dog" / "dog-1.png", (40, 40, 200))
    checkpoint_before = (experiment / "best_model.pth").read_bytes()
    metrics_before = (experiment / "metrics.json").read_bytes()

    result = evaluate_model_asset(
        root,
        "phase6-model",
        dataset,
        batch_size=1,
        device="cpu",
    )

    assert result.metrics["sample_count"] == 2
    assert result.metrics["confusion_matrix"] == [[1, 0], [1, 0]]
    assert result.sample_ids == ["cat/cat-1.png", "dog/dog-1.png"]
    assert result.metadata["asset_id"] == "phase6-model"
    assert (result.artifacts["predictions"]).is_file()
    with result.artifacts["predictions"].open(encoding="utf-8", newline="") as source:
        rows = list(csv.DictReader(source))
    assert rows[1]["sample_id"] == "dog/dog-1.png"
    assert rows[1]["relative_path"] == "dog/dog-1.png"
    assert rows[1]["score_cat"] == "2.0"
    assert result.artifacts["roc_curve"].is_file()
    assert result.artifacts["precision_recall_curve"].is_file()
    assert (experiment / "best_model.pth").read_bytes() == checkpoint_before
    assert (experiment / "metrics.json").read_bytes() == metrics_before

    monkeypatch.setattr(app, "OUTPUT_ROOT", root)
    ui_result = app.run_asset_evaluation_ui(
        "phase6-model", str(dataset), "image_folder", None, 1, 0.5
    )
    state, status, report = ui_result[:3]
    assert "评估完成" in status
    assert report["metrics"]["sample_count"] == 2
    error_frame, choices = app.filter_evaluation_errors(state, "dog")
    assert len(error_frame) == 1
    sample_id = choices["choices"][0]
    image_path, details = app.preview_evaluation_error(sample_id, state)
    assert image_path.endswith("dog-1.png")
    assert "dog" in details

    single = predict_model_asset_image(
        root, "phase6-model", dataset / "cat" / "cat-1.png", device="cpu"
    )
    assert single["prediction"] == "cat"
    assert single["scores"]["cat"] == 2.0
    assert sum(single["probabilities"].values()) == pytest.approx(1.0)

    image_directory = tmp_path / "unlabelled"
    _write_image(image_directory / "nested" / "z.png", (80, 80, 80))
    _write_image(image_directory / "a.png", (20, 20, 20))
    predictions, exported_csv = predict_model_asset_directory(
        root,
        "phase6-model",
        image_directory,
        batch_size=1,
        device="cpu",
    )
    assert predictions.sample_ids == ["a.png", "nested/z.png"]
    with exported_csv.open(encoding="utf-8", newline="") as source:
        batch_rows = list(csv.DictReader(source))
    assert [row["relative_path"] for row in batch_rows] == ["a.png", "nested/z.png"]

    exported_weights = export_model_asset(root, "phase6-model")
    reloaded = load_exported_model_asset(exported_weights)
    assert reloaded["configuration"]["dataset"]["image_size"] == 4
    assert reloaded["configuration"]["dataset"]["mean"] == [0.5, 0.5, 0.5]
    assert reloaded["configuration"]["model"]["pretrained"] is True
    assert reloaded["asset"]["class_names"] == ["cat", "dog"]
    torch.testing.assert_close(
        reloaded["model"](torch.zeros(1, 3, 4, 4)),
        torch.tensor([[2.0, -2.0]]),
    )
    ui_weights, ui_sidecar, export_status = app.export_model_asset_ui("phase6-model")
    assert Path(ui_weights).is_file()
    assert Path(ui_sidecar).is_file()
    assert "结构/预处理配置" in export_status


def test_multilabel_csv_evaluation_and_inference_use_label_thresholds(
    tmp_path: Path, tiny_model: None
) -> None:
    root = tmp_path / "runs"
    _make_asset(
        root,
        task="multi_label",
        data_format="csv_multilabel",
        labels=["red", "round"],
    )
    dataset = tmp_path / "multilabel"
    _write_image(dataset / "images" / "one.png", (150, 20, 20))
    _write_image(dataset / "images" / "two.png", (20, 20, 150))
    (dataset / "annotations.csv").write_text(
        "image_path,red,round\none.png,1,0\ntwo.png,0,1\n", encoding="utf-8"
    )

    result = evaluate_model_asset(
        root,
        "phase6-model",
        dataset,
        data_format="csv_multilabel",
        batch_size=2,
        device="cpu",
    )

    assert result.task == "multi_label"
    assert result.metrics["per_class"]["round"]["positive_support"] == 1
    assert result.metrics["per_class"]["round"]["average_precision"] is not None
    assert result.predictions.tolist() == [[1, 0], [1, 0]]
    assert result.warnings == []


def test_csv_evaluation_retains_absent_model_classes_and_reports_undefined_curves(
    tmp_path: Path, tiny_model: None
) -> None:
    root = tmp_path / "runs"
    _make_asset(root)
    dataset = tmp_path / "csv-evaluation"
    _write_image(dataset / "images" / "only-cat.png", (180, 30, 30))
    (dataset / "annotations.csv").write_text(
        "image_path,label\nonly-cat.png,cat\n", encoding="utf-8"
    )

    result = evaluate_model_asset(
        root,
        "phase6-model",
        dataset / "annotations.csv",
        data_format="csv_classification",
        image_root=dataset / "images",
        device="cpu",
    )

    assert result.metrics["per_class"]["dog"]["support"] == 0
    assert result.metrics["per_class"]["dog"]["auc"] is None
    assert result.metrics["per_class"]["dog"]["average_precision"] is None
    assert any("dog: ROC-AUC" in warning for warning in result.warnings)
    assert result.class_names == ["cat", "dog"]

    with pytest.raises(ValueError, match="does not match task"):
        evaluate_model_asset(
            root,
            "phase6-model",
            dataset,
            data_format="csv_multilabel",
            device="cpu",
        )


def test_historical_experiment_checkpoint_can_be_evaluated_without_registration(
    tmp_path: Path, tiny_model: None
) -> None:
    root = tmp_path / "historical-runs"
    experiment = root / "experiment_legacy"
    experiment.mkdir(parents=True)
    configuration = {
        "task": {"type": "classification"},
        "dataset": {
            "format": "image_folder",
            "image_size": 4,
            "mean": [0.5, 0.5, 0.5],
            "std": [0.25, 0.25, 0.25],
        },
        "model": {"name": "phase6_tiny", "num_classes": 2, "pretrained": True},
    }
    (experiment / "config.yaml").write_text(yaml.safe_dump(configuration), encoding="utf-8")
    model = _builder(2, False)
    with torch.no_grad():
        model[1].weight.zero_()
        model[1].bias.copy_(torch.tensor([2.0, -2.0]))
    torch.save({"model_state_dict": model.state_dict()}, experiment / "best_model.pth")
    (experiment / "experiment.json").write_text(
        json.dumps(
            {
                "id": "historical-run",
                "task": "classification",
                "dataset": {"class_names": ["cat", "dog"]},
            }
        ),
        encoding="utf-8",
    )
    dataset = tmp_path / "holdout"
    _write_image(dataset / "cat" / "sample.png", (170, 30, 30))
    _write_image(dataset / "dog" / "sample.png", (30, 30, 170))

    assets = list_model_assets(root)
    result = evaluate_model_asset(
        root,
        "legacy-model-historical-run",
        dataset,
        device="cpu",
    )

    assert assets[0]["legacy"] is True
    assert assets[0]["class_names"] == ["cat", "dog"]
    assert result.metadata["experiment_id"] == "historical-run"
    assert result.metrics["sample_count"] == 2
