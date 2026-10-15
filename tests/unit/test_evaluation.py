from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from uatf.config import validate_config
from uatf.evaluation import (
    CUSTOM_METRIC_REGISTRY,
    classification_metrics,
    evaluate_model,
    evaluate_training_run,
    metric_names_for_task,
    multilabel_metrics,
    register_metric,
    save_evaluation_plots,
)


def test_classification_metrics_include_macro_micro_and_missing_class() -> None:
    result = classification_metrics(
        targets=[0, 0, 1, 1],
        predictions=[0, 1, 1, 1],
        class_names=["Disease A", "Disease B", "Disease C"],
    )

    assert result["accuracy"] == pytest.approx(0.75)
    assert result["precision"] == pytest.approx((1 + 2 / 3 + 0) / 3)
    assert result["recall"] == pytest.approx((0.5 + 1 + 0) / 3)
    assert result["f1"] == pytest.approx((2 / 3 + 0.8 + 0) / 3)
    assert result["micro_f1"] == pytest.approx(0.75)
    assert result["confusion_matrix"] == [[1, 1, 0], [0, 2, 0], [0, 0, 0]]
    assert result["per_class"]["Disease C"]["support"] == 0
    assert result["per_class"]["Disease C"]["precision"] == 0


def test_registered_metric_is_added_to_task_compatible_evaluation_and_config() -> None:
    name = "exact_match_test"
    register_metric(
        name,
        lambda targets, predictions: (targets == predictions).float().mean(),
        supported_tasks=("classification",),
    )
    try:
        assert name in metric_names_for_task("classification")
        assert name not in metric_names_for_task("multi_label")
        result = classification_metrics([0, 1], [0, 0])
        assert result[name] == pytest.approx(0.5)
        config = validate_config(
            {
                "dataset": {"path": "unused-dataset"},
                "evaluation": {"primary_metric": name},
            }
        )
        assert config["evaluation"]["primary_metric"] == name
    finally:
        CUSTOM_METRIC_REGISTRY.pop(name, None)


def test_multilabel_auc_per_class_macro_and_missing_positive_class() -> None:
    targets = [
        [0, 0, 0],
        [0, 1, 0],
        [1, 0, 0],
        [1, 1, 0],
    ]
    probabilities = [
        [0.1, 0.1, 0.1],
        [0.4, 0.9, 0.2],
        [0.35, 0.2, 0.3],
        [0.8, 0.8, 0.4],
    ]

    result = multilabel_metrics(
        targets,
        probabilities,
        class_names=["Disease A", "Disease B", "Disease C"],
    )

    assert result["per_class"]["Disease A"]["auc"] == pytest.approx(0.75)
    assert result["per_class"]["Disease B"]["auc"] == pytest.approx(1.0)
    assert result["per_class"]["Disease C"]["auc"] is None
    assert result["macro_auc"] == pytest.approx(0.875)
    assert result["macro_auc_class_count"] == 2
    assert result["per_class"]["Disease C"]["positive_support"] == 0
    assert result["roc_curves"]["Disease C"]["false_positive_rate"] == []


def test_nan_and_invalid_multilabel_targets_are_rejected() -> None:
    with pytest.raises(ValueError, match="NaN or infinity"):
        classification_metrics([0, float("nan")], [0, 1])
    with pytest.raises(ValueError, match="only 0 and 1"):
        multilabel_metrics([[0, 0.5]], [[0.2, 0.8]])


def test_auc_treats_tied_positive_and_negative_scores_as_half_correct() -> None:
    result = multilabel_metrics([[0], [1]], [[0.5], [0.5]])

    assert result["per_class"]["class_0"]["auc"] == pytest.approx(0.5)


def test_evaluate_model_aggregates_classification_batches() -> None:
    logits = torch.tensor([[3.0, 0.0], [0.0, 3.0], [0.0, 3.0], [3.0, 0.0]])
    targets = torch.tensor([0, 0, 1, 1])
    result = evaluate_model(
        nn.Identity(),
        DataLoader(TensorDataset(logits, targets), batch_size=2),
        task="classification",
        class_names=["negative", "positive"],
    )

    assert result.metrics["accuracy"] == pytest.approx(0.5)
    assert result.metrics["precision"] == pytest.approx(0.5)
    assert result.metrics["recall"] == pytest.approx(0.5)
    assert result.metrics["f1"] == pytest.approx(0.5)
    assert result.metrics["confusion_matrix"] == [[1, 1], [1, 1]]


def test_evaluate_training_run_writes_metrics_and_predictions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("uatf.evaluation.evaluator.save_evaluation_plots", lambda *args, **kwargs: {})
    model = nn.Identity()
    torch.save({"model_state_dict": model.state_dict()}, tmp_path / "best_model.pth")
    logits = torch.tensor([[3.0, 0.0], [0.0, 3.0], [0.0, 3.0], [3.0, 0.0]])
    targets = torch.tensor([0, 0, 1, 1])
    loader = DataLoader(TensorDataset(logits, targets), batch_size=2)

    result = evaluate_training_run(
        model,
        loader,
        tmp_path,
        task="classification",
        class_names=["negative", "positive"],
        history=[
            {"epoch": 1, "train_loss": 0.8, "val_loss": 0.7, "val_accuracy": 0.5},
            {"epoch": 2, "train_loss": 0.5, "val_loss": 0.6, "val_accuracy": 0.75},
        ],
    )

    assert result.metrics["accuracy"] == pytest.approx(0.5)
    assert result.metrics["precision"] == pytest.approx(0.5)
    assert result.metrics["recall"] == pytest.approx(0.5)
    assert result.metrics["f1"] == pytest.approx(0.5)
    assert result.metrics["confusion_matrix"] == [[1, 1], [1, 1]]

    report = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
    assert report["metrics"]["accuracy"] == pytest.approx(0.5)
    assert (tmp_path / "predictions.csv").is_file()
    with (tmp_path / "predictions.csv").open(encoding="utf-8", newline="") as source:
        rows = list(csv.DictReader(source))
    assert len(rows) == 4
    assert rows[0]["target"] == "negative"


def _multilabel_fixture() -> tuple[torch.Tensor, torch.Tensor]:
    probabilities = torch.tensor(
        [
            [0.1, 0.1, 0.1],
            [0.4, 0.9, 0.2],
            [0.35, 0.2, 0.3],
            [0.8, 0.8, 0.4],
        ]
    )
    targets = torch.tensor([[0, 0, 0], [0, 1, 0], [1, 0, 0], [1, 1, 0]], dtype=torch.float32)
    return torch.logit(probabilities), targets


def test_evaluate_model_aggregates_multilabel_batches() -> None:
    class Logits(nn.Module):
        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            return inputs

    logits, targets = _multilabel_fixture()
    result = evaluate_model(
        Logits(),
        DataLoader(TensorDataset(logits, targets), batch_size=3),
        task="multi_label",
        class_names=["Disease A", "Disease B", "Disease C"],
    )

    assert result.metrics["macro_auc"] == pytest.approx(0.875)
    assert result.metrics["per_class"]["Disease C"]["auc"] is None


def test_multilabel_training_run_writes_auc_report_and_predictions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Logits(nn.Module):
        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            return inputs

    monkeypatch.setattr("uatf.evaluation.evaluator.save_evaluation_plots", lambda *args, **kwargs: {})
    model = Logits()
    logits, targets = _multilabel_fixture()
    torch.save({"model_state_dict": model.state_dict()}, tmp_path / "best_model.pth")
    loader = DataLoader(TensorDataset(logits, targets), batch_size=3)

    result = evaluate_training_run(
        model,
        loader,
        tmp_path,
        task="multi_label",
        class_names=["Disease A", "Disease B", "Disease C"],
    )

    assert result.metrics["macro_auc"] == pytest.approx(0.875)
    report = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
    assert report["metrics"]["per_class"]["Disease C"]["auc"] is None


@pytest.mark.skipif(
    importlib.util.find_spec("matplotlib") is None,
    reason="matplotlib is not installed in the active test environment",
)
def test_visualization_writes_confusion_and_per_class_roc_plots(tmp_path: Path) -> None:
    class Logits(nn.Module):
        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            return inputs

    classification_logits = torch.tensor([[3.0, 0.0], [0.0, 3.0]])
    classification_targets = torch.tensor([0, 1])
    classification = evaluate_model(
        nn.Identity(),
        DataLoader(TensorDataset(classification_logits, classification_targets), batch_size=2),
        task="classification",
        class_names=["negative", "positive"],
    )
    multilabel_logits, multilabel_targets = _multilabel_fixture()
    multilabel = evaluate_model(
        Logits(),
        DataLoader(TensorDataset(multilabel_logits, multilabel_targets), batch_size=3),
        task="multi_label",
        class_names=["Disease A", "Disease B", "Disease C"],
    )

    classification_paths = save_evaluation_plots(
        classification,
        tmp_path / "classification",
        history=[{"epoch": 1, "train_loss": 1.0, "val_loss": 0.5, "val_accuracy": 1.0}],
    )
    multilabel_paths = save_evaluation_plots(multilabel, tmp_path / "multilabel")

    assert classification_paths["confusion_matrix"].is_file()
    assert classification_paths["loss_curve"].is_file()
    assert classification_paths["metric_curve"].is_file()
    assert multilabel_paths["roc_curve"].is_file()
    assert multilabel_paths["roc_Disease A"].is_file()
    assert multilabel_paths["roc_Disease C"].is_file()
