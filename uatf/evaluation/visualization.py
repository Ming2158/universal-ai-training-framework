"""Create loss, metric, confusion-matrix, and ROC plots."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from uatf.evaluation.evaluator import EvaluationResult


def _pyplot():
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    return plt


def _save_training_curves(
    history: Sequence[Mapping[str, float]] | None,
    output_dir: Path,
) -> dict[str, Path]:
    plt = _pyplot()
    rows = list(history or [])
    epochs = [row.get("epoch", index + 1) for index, row in enumerate(rows)]

    loss_path = output_dir / "loss_curve.png"
    figure, axis = plt.subplots(figsize=(8, 5))
    for key, label, color in (
        ("train_loss", "Training loss", "tab:blue"),
        ("val_loss", "Validation loss", "tab:orange"),
    ):
        values = [row.get(key) for row in rows]
        valid = [(epoch, value) for epoch, value in zip(epochs, values, strict=True) if value is not None]
        if valid:
            axis.plot(
                [item[0] for item in valid],
                [item[1] for item in valid],
                marker="o",
                label=label,
                color=color,
            )
    if rows and (any("train_loss" in row for row in rows) or any("val_loss" in row for row in rows)):
        axis.set_xlabel("Epoch")
        axis.set_ylabel("Loss")
        axis.legend()
    else:
        axis.text(0.5, 0.5, "No training history supplied", ha="center", va="center")
        axis.set_axis_off()
    axis.set_title("Training and validation loss")
    axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(loss_path, dpi=150)
    plt.close(figure)

    metric_path = output_dir / "metric_curve.png"
    figure, axis = plt.subplots(figsize=(8, 5))
    metric_keys = sorted(
        {
            key
            for row in rows
            for key in row
            if key.startswith(("train_", "val_")) and key not in {"train_loss", "val_loss"}
        }
    )
    for key in metric_keys:
        values = [row.get(key) for row in rows]
        valid = [
            (epoch, value)
            for epoch, value in zip(epochs, values, strict=True)
            if value is not None
        ]
        if valid:
            axis.plot(
                [item[0] for item in valid],
                [item[1] for item in valid],
                marker="o",
                label=key.replace("_", " "),
            )
    if metric_keys:
        axis.set_xlabel("Epoch")
        axis.set_ylabel("Metric")
        axis.legend()
    else:
        axis.text(0.5, 0.5, "No metric series in training history", ha="center", va="center")
        axis.set_axis_off()
    axis.set_title("Training metrics")
    axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(metric_path, dpi=150)
    plt.close(figure)
    return {"loss_curve": loss_path, "metric_curve": metric_path}


def _save_confusion_matrix(result: EvaluationResult, output_dir: Path) -> Path:
    plt = _pyplot()
    matrix = result.metrics["confusion_matrix"]
    labels = result.metrics["confusion_matrix_labels"]
    figure_size = max(6, min(12, len(labels) * 0.7))
    figure, axis = plt.subplots(figsize=(figure_size, figure_size))
    image = axis.imshow(matrix, cmap="Blues")
    figure.colorbar(image, ax=axis, label="Samples")
    axis.set_xticks(range(len(labels)), labels=labels, rotation=45, ha="right")
    axis.set_yticks(range(len(labels)), labels=labels)
    axis.set_xlabel("Predicted class")
    axis.set_ylabel("True class")
    axis.set_title("Confusion matrix")
    midpoint = max((max(row) for row in matrix), default=0) / 2
    for row_index, row in enumerate(matrix):
        for column_index, value in enumerate(row):
            color = "white" if value > midpoint and midpoint > 0 else "black"
            axis.text(column_index, row_index, str(value), ha="center", va="center", color=color)
    figure.tight_layout()
    path = output_dir / "confusion_matrix.png"
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def _save_roc_plot(
    curve: Mapping[str, Any],
    output_path: Path,
    *,
    title: str,
    label: str,
) -> None:
    plt = _pyplot()
    figure, axis = plt.subplots(figsize=(7, 6))
    auc = curve.get("auc")
    fpr = curve.get("false_positive_rate", [])
    tpr = curve.get("true_positive_rate", [])
    if auc is None or not fpr:
        axis.text(
            0.5,
            0.5,
            "ROC-AUC unavailable: both positive and negative samples are required",
            ha="center",
            va="center",
            wrap=True,
        )
    else:
        axis.plot(fpr, tpr, label=f"{label} (AUC = {auc:.3f})", color="tab:blue")
        axis.plot([0, 1], [0, 1], linestyle="--", color="tab:gray", label="Random baseline")
        axis.legend(loc="lower right")
    axis.set_xlabel("False positive rate")
    axis.set_ylabel("True positive rate")
    axis.set_title(title)
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1.02)
    axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def _save_multilabel_roc(
    result: EvaluationResult,
    output_dir: Path,
) -> dict[str, Path]:
    plt = _pyplot()
    paths: dict[str, Path] = {}
    combined_path = output_dir / "roc_curve.png"
    figure, axis = plt.subplots(figsize=(8, 7))
    for index, (name, curve) in enumerate(result.roc_curves.items()):
        auc = curve.get("auc")
        fpr = curve.get("false_positive_rate", [])
        tpr = curve.get("true_positive_rate", [])
        if auc is not None and fpr:
            axis.plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})")
        slug = re.sub(r"[^A-Za-z0-9_-]+", "_", name).strip("_") or "label"
        class_path = output_dir / f"roc_{index:02d}_{slug}.png"
        _save_roc_plot(
            curve,
            class_path,
            title=f"ROC curve: {name}",
            label=name,
        )
        paths[f"roc_{name}"] = class_path

    if any(curve.get("auc") is not None for curve in result.roc_curves.values()):
        axis.plot([0, 1], [0, 1], linestyle="--", color="tab:gray", label="Random baseline")
        axis.legend(loc="lower right")
    else:
        axis.text(
            0.5,
            0.5,
            "ROC-AUC unavailable: no label has both positive and negative samples",
            ha="center",
            va="center",
            wrap=True,
        )
    axis.set_xlabel("False positive rate")
    axis.set_ylabel("True positive rate")
    axis.set_title("Multi-label ROC curves")
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1.02)
    axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(combined_path, dpi=150)
    plt.close(figure)
    paths["roc_curve"] = combined_path
    return paths


def save_evaluation_plots(
    result: EvaluationResult,
    output_dir: str | Path,
    *,
    history: Sequence[Mapping[str, float]] | None = None,
) -> dict[str, Path]:
    """Write the plots appropriate to an evaluation task and available history."""
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    paths = _save_training_curves(history, destination)
    if result.task == "classification":
        paths["confusion_matrix"] = _save_confusion_matrix(result, destination)
    else:
        paths.update(_save_multilabel_roc(result, destination))
    return paths
