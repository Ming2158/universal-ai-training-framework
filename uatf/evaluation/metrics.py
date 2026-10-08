"""Reliable metrics for single-label and multi-label classification."""

from __future__ import annotations

from collections.abc import Sequence
from math import isfinite
from typing import Any

import torch
from torch import Tensor


def _tensor(value: Any, name: str) -> Tensor:
    try:
        result = torch.as_tensor(value).detach().cpu()
    except (TypeError, ValueError, RuntimeError) as error:
        raise ValueError(f"{name} must be convertible to a tensor.") from error
    if result.numel() == 0:
        raise ValueError(f"{name} cannot be empty.")
    if not torch.isfinite(result.to(torch.float64)).all().item():
        raise ValueError(f"{name} contains NaN or infinity.")
    return result


def _class_names(class_names: Sequence[str] | None, count: int) -> list[str]:
    if class_names is None:
        return [f"class_{index}" for index in range(count)]
    names = list(class_names)
    if len(names) != count or any(not isinstance(name, str) or not name for name in names):
        raise ValueError(f"class_names must contain exactly {count} non-empty strings.")
    if len(set(names)) != len(names):
        raise ValueError("class_names must be unique.")
    return names


def _safe_divide(numerator: Tensor, denominator: Tensor) -> Tensor:
    smallest = torch.finfo(denominator.dtype).tiny
    return torch.where(denominator > 0, numerator / denominator.clamp_min(smallest), 0.0)


def _f1(precision: Tensor, recall: Tensor) -> Tensor:
    return _safe_divide(2 * precision * recall, precision + recall)


def _roc_curve(targets: Tensor, scores: Tensor) -> dict[str, Any]:
    """Build a tie-aware ROC curve; AUC is undefined without both target values."""
    positives = int(targets.sum().item())
    negatives = targets.numel() - positives
    if positives == 0 or negatives == 0:
        return {"auc": None, "false_positive_rate": [], "true_positive_rate": []}

    order = torch.argsort(scores, descending=True, stable=True)
    sorted_scores = scores[order]
    sorted_targets = targets[order].to(torch.float64)
    true_positives = sorted_targets.cumsum(dim=0)
    false_positives = torch.arange(1, targets.numel() + 1, dtype=torch.float64) - true_positives
    group_ends = torch.ones(targets.numel(), dtype=torch.bool)
    group_ends[:-1] = sorted_scores[:-1] != sorted_scores[1:]

    fpr = torch.cat((torch.zeros(1, dtype=torch.float64), false_positives[group_ends] / negatives))
    tpr = torch.cat((torch.zeros(1, dtype=torch.float64), true_positives[group_ends] / positives))
    widths = fpr[1:] - fpr[:-1]
    auc = float((widths * (tpr[1:] + tpr[:-1]) / 2).sum().item())
    return {
        "auc": auc,
        "false_positive_rate": fpr.tolist(),
        "true_positive_rate": tpr.tolist(),
    }


def classification_metrics(
    targets: Any,
    predictions: Any,
    *,
    num_classes: int | None = None,
    class_names: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Calculate multiclass metrics from integer class IDs and predictions.

    Per-class values use zero when a denominator is zero. ``num_classes`` or
    ``class_names`` can retain classes that are absent from this evaluation split.
    """
    target_values = _tensor(targets, "targets")
    prediction_values = _tensor(predictions, "predictions")
    if target_values.ndim != 1 or prediction_values.ndim != 1:
        raise ValueError("Classification targets and predictions must be one-dimensional class IDs.")
    if target_values.shape != prediction_values.shape:
        raise ValueError("Classification targets and predictions must have the same shape.")
    if not torch.equal(target_values, target_values.round()) or not torch.equal(
        prediction_values, prediction_values.round()
    ):
        raise ValueError("Classification targets and predictions must be integer class IDs.")
    target_values = target_values.to(torch.long)
    prediction_values = prediction_values.to(torch.long)
    if target_values.min().item() < 0 or prediction_values.min().item() < 0:
        raise ValueError("Classification class IDs cannot be negative.")

    inferred_count = int(max(target_values.max().item(), prediction_values.max().item())) + 1
    if class_names is not None:
        if num_classes is not None and num_classes != len(class_names):
            raise ValueError("num_classes and class_names have different lengths.")
        num_classes = len(class_names)
    if num_classes is None:
        num_classes = inferred_count
    if isinstance(num_classes, bool) or not isinstance(num_classes, int) or num_classes < 1:
        raise ValueError("num_classes must be a positive integer.")
    if inferred_count > num_classes:
        raise ValueError("A target or prediction contains a class ID outside num_classes.")
    names = _class_names(class_names, num_classes)

    encoded = target_values * num_classes + prediction_values
    matrix = torch.bincount(encoded, minlength=num_classes * num_classes).reshape(
        num_classes, num_classes
    )
    true_positive = matrix.diag().to(torch.float64)
    support = matrix.sum(dim=1).to(torch.float64)
    predicted_count = matrix.sum(dim=0).to(torch.float64)
    precision = _safe_divide(true_positive, predicted_count)
    recall = _safe_divide(true_positive, support)
    f1 = _f1(precision, recall)
    correct = int(true_positive.sum().item())
    sample_count = target_values.numel()
    micro_precision = float(correct / sample_count)
    per_class = {
        name: {
            "precision": float(precision[index].item()),
            "recall": float(recall[index].item()),
            "f1": float(f1[index].item()),
            "support": int(support[index].item()),
        }
        for index, name in enumerate(names)
    }
    macro_precision = float(precision.mean().item())
    macro_recall = float(recall.mean().item())
    macro_f1 = float(f1.mean().item())
    return {
        "task": "classification",
        "sample_count": sample_count,
        "accuracy": correct / sample_count,
        "precision": macro_precision,
        "recall": macro_recall,
        "f1": macro_f1,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "micro_precision": micro_precision,
        "micro_recall": micro_precision,
        "micro_f1": micro_precision,
        "per_class": per_class,
        "confusion_matrix": matrix.tolist(),
        "confusion_matrix_labels": names,
    }


def multilabel_metrics(
    targets: Any,
    probabilities: Any,
    *,
    class_names: Sequence[str] | None = None,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Calculate multilabel metrics from an ``N x C`` target/probability matrix.

    AUC values are ``None`` for labels whose evaluation targets contain only one
    value. Macro AUC averages only the labels with a defined AUC.
    """
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not isfinite(threshold):
        raise ValueError("threshold must be a finite number between 0 and 1.")
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be a finite number between 0 and 1.")

    target_values = _tensor(targets, "targets").to(torch.float64)
    probability_values = _tensor(probabilities, "probabilities").to(torch.float64)
    if target_values.ndim != 2 or probability_values.ndim != 2:
        raise ValueError("Multi-label targets and probabilities must be two-dimensional (N x C).")
    if target_values.shape != probability_values.shape:
        raise ValueError("Multi-label targets and probabilities must have the same shape.")
    if not torch.logical_or(target_values == 0, target_values == 1).all().item():
        raise ValueError("Multi-label targets must contain only 0 and 1.")
    if ((probability_values < 0) | (probability_values > 1)).any().item():
        raise ValueError("Multi-label probabilities must be between 0 and 1.")

    label_count = target_values.shape[1]
    if label_count < 1:
        raise ValueError("Multi-label evaluation requires at least one label.")
    names = _class_names(class_names, label_count)
    binary_predictions = (probability_values >= threshold).to(torch.float64)
    true_positive = (target_values * binary_predictions).sum(dim=0)
    false_positive = ((1 - target_values) * binary_predictions).sum(dim=0)
    false_negative = (target_values * (1 - binary_predictions)).sum(dim=0)
    precision = _safe_divide(true_positive, true_positive + false_positive)
    recall = _safe_divide(true_positive, true_positive + false_negative)
    f1 = _f1(precision, recall)

    roc_curves = {
        name: _roc_curve(target_values[:, index].to(torch.bool), probability_values[:, index])
        for index, name in enumerate(names)
    }
    valid_auc = [curve["auc"] for curve in roc_curves.values() if curve["auc"] is not None]
    micro_roc = _roc_curve(target_values.flatten().to(torch.bool), probability_values.flatten())
    total_tp = true_positive.sum()
    total_fp = false_positive.sum()
    total_fn = false_negative.sum()
    micro_precision = float(_safe_divide(total_tp, total_tp + total_fp).item())
    micro_recall = float(_safe_divide(total_tp, total_tp + total_fn).item())
    micro_f1 = float(
        _f1(
            torch.tensor(micro_precision, dtype=torch.float64),
            torch.tensor(micro_recall, dtype=torch.float64),
        ).item()
    )
    per_class = {
        name: {
            "precision": float(precision[index].item()),
            "recall": float(recall[index].item()),
            "f1": float(f1[index].item()),
            "auc": roc_curves[name]["auc"],
            "positive_support": int(target_values[:, index].sum().item()),
            "negative_support": int((1 - target_values[:, index]).sum().item()),
        }
        for index, name in enumerate(names)
    }
    macro_precision = float(precision.mean().item())
    macro_recall = float(recall.mean().item())
    macro_f1 = float(f1.mean().item())
    return {
        "task": "multi_label",
        "sample_count": target_values.shape[0],
        "threshold": float(threshold),
        "precision": macro_precision,
        "recall": macro_recall,
        "f1": macro_f1,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "micro_precision": micro_precision,
        "micro_recall": micro_recall,
        "micro_f1": micro_f1,
        "macro_auc": sum(valid_auc) / len(valid_auc) if valid_auc else None,
        "macro_auc_class_count": len(valid_auc),
        "micro_auc": micro_roc["auc"],
        "per_class": per_class,
        "roc_curves": roc_curves,
        "micro_roc_curve": micro_roc,
    }
