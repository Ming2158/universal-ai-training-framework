"""Run whole-dataset evaluation and write experiment artifacts."""

from __future__ import annotations

import csv
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from uatf.evaluation.metrics import classification_metrics, multilabel_metrics
from uatf.evaluation.visualization import save_evaluation_plots
from uatf.models.factory import normalize_task


@dataclass
class EvaluationResult:
    """Predictions and aggregate metrics from one complete evaluation pass."""

    task: str
    class_names: list[str]
    targets: Tensor
    predictions: Tensor
    probabilities: Tensor
    metrics: dict[str, Any]
    roc_curves: dict[str, dict[str, Any]]
    loss: float | None = None


def _infer_class_names(loader: Iterable[Any]) -> list[str] | None:
    dataset = getattr(loader, "dataset", None)
    while dataset is not None:
        names = getattr(dataset, "classes", None)
        if names is not None:
            return list(names)
        dataset = getattr(dataset, "dataset", None)
    return None


def _model_device(model: nn.Module, requested: str | torch.device | None) -> torch.device:
    if requested is not None:
        if requested == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(requested)
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def evaluate_model(
    model: nn.Module,
    data_loader: Iterable[tuple[Tensor, Tensor]],
    *,
    task: str,
    class_names: Sequence[str] | None = None,
    threshold: float = 0.5,
    device: str | torch.device | None = None,
    loss_function: nn.Module | None = None,
) -> EvaluationResult:
    """Evaluate all batches together so global metrics and ROC-AUC are correct."""
    normalized_task = normalize_task(task)
    names = list(class_names) if class_names is not None else _infer_class_names(data_loader)
    resolved_device = _model_device(model, device)
    all_targets: list[Tensor] = []
    all_logits: list[Tensor] = []
    weighted_loss = 0.0
    sample_count = 0

    model.to(resolved_device)
    model.eval()
    with torch.no_grad():
        for batch_index, batch in enumerate(data_loader, start=1):
            if not isinstance(batch, (tuple, list)) or len(batch) != 2:
                raise ValueError(f"Evaluation batch {batch_index} must be (inputs, targets).")
            inputs, targets = batch
            if not isinstance(inputs, Tensor) or not isinstance(targets, Tensor):
                raise ValueError(f"Evaluation batch {batch_index} must contain tensors.")
            if inputs.ndim == 0 or targets.ndim == 0 or inputs.shape[0] != targets.shape[0]:
                raise ValueError(f"Evaluation batch {batch_index} has incompatible batch shapes.")
            logits = model(inputs.to(resolved_device))
            if not isinstance(logits, Tensor) or logits.ndim != 2:
                raise ValueError("Evaluation models must return a two-dimensional (N x C) tensor.")
            targets = targets.to(resolved_device)
            if logits.shape[0] != targets.shape[0]:
                raise ValueError(f"Evaluation batch {batch_index} has incompatible output and target sizes.")
            if not torch.isfinite(logits).all().item():
                raise FloatingPointError(f"Model output contains NaN or infinity in batch {batch_index}.")

            if normalized_task == "classification":
                if targets.ndim != 1:
                    raise ValueError("Classification targets must be one-dimensional class IDs.")
            elif targets.ndim != 2 or targets.shape != logits.shape:
                raise ValueError("Multi-label targets must have the same (N x C) shape as model outputs.")

            if loss_function is not None:
                loss = loss_function(logits, targets)
                if loss.numel() != 1 or not torch.isfinite(loss).all().item():
                    raise FloatingPointError(f"Evaluation loss is invalid in batch {batch_index}.")
                loss_scale = 1 if getattr(loss_function, "reduction", "mean") == "sum" else targets.shape[0]
                weighted_loss += float(loss.item()) * loss_scale
            all_targets.append(targets.detach().cpu())
            all_logits.append(logits.detach().cpu())
            sample_count += targets.shape[0]

    if sample_count == 0:
        raise ValueError("Evaluation DataLoader yielded no examples.")
    target_values = torch.cat(all_targets, dim=0)
    logits_values = torch.cat(all_logits, dim=0)

    if normalized_task == "classification":
        probabilities = torch.softmax(logits_values, dim=1)
        predictions = probabilities.argmax(dim=1)
        metrics = classification_metrics(
            target_values,
            predictions,
            num_classes=logits_values.shape[1],
            class_names=names,
        )
        roc_curves: dict[str, dict[str, Any]] = {}
        resolved_names = metrics["confusion_matrix_labels"]
    else:
        probabilities = torch.sigmoid(logits_values)
        metrics = multilabel_metrics(
            target_values,
            probabilities,
            class_names=names,
            threshold=threshold,
        )
        predictions = (probabilities >= threshold).to(torch.int64)
        roc_curves = metrics.pop("roc_curves")
        metrics.pop("micro_roc_curve")
        resolved_names = list(names) if names is not None else [
            f"class_{index}" for index in range(logits_values.shape[1])
        ]

    return EvaluationResult(
        task=normalized_task,
        class_names=resolved_names,
        targets=target_values,
        predictions=predictions,
        probabilities=probabilities,
        metrics=metrics,
        roc_curves=roc_curves,
        loss=weighted_loss / sample_count if loss_function is not None else None,
    )


def _write_predictions(result: EvaluationResult, path: Path) -> None:
    target_rows = result.targets.tolist()
    prediction_rows = result.predictions.tolist()
    probability_rows = result.probabilities.tolist()
    if result.task == "classification":
        columns = ["sample_index", "target", "prediction", "confidence"]
        columns.extend(f"probability_{name}" for name in result.class_names)
        with path.open("w", encoding="utf-8", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=columns)
            writer.writeheader()
            for index, (target, prediction, probabilities) in enumerate(
                zip(target_rows, prediction_rows, probability_rows, strict=True)
            ):
                row = {
                    "sample_index": index,
                    "target": result.class_names[int(target)],
                    "prediction": result.class_names[int(prediction)],
                    "confidence": max(probabilities),
                }
                row.update(
                    {
                        f"probability_{name}": probability
                        for name, probability in zip(result.class_names, probabilities, strict=True)
                    }
                )
                writer.writerow(row)
        return

    columns = ["sample_index"]
    columns.extend(f"target_{name}" for name in result.class_names)
    columns.extend(f"prediction_{name}" for name in result.class_names)
    columns.extend(f"probability_{name}" for name in result.class_names)
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=columns)
        writer.writeheader()
        for index, (targets, predictions, probabilities) in enumerate(
            zip(target_rows, prediction_rows, probability_rows, strict=True)
        ):
            row: dict[str, Any] = {"sample_index": index}
            row.update(
                {f"target_{name}": int(value) for name, value in zip(result.class_names, targets, strict=True)}
            )
            row.update(
                {
                    f"prediction_{name}": int(value)
                    for name, value in zip(result.class_names, predictions, strict=True)
                }
            )
            row.update(
                {
                    f"probability_{name}": value
                    for name, value in zip(result.class_names, probabilities, strict=True)
                }
            )
            writer.writerow(row)


def save_evaluation_artifacts(
    result: EvaluationResult,
    output_dir: str | Path,
    *,
    history: Sequence[Mapping[str, float]] | None = None,
) -> dict[str, Path]:
    """Write metrics, predictions, and plots under one experiment directory."""
    destination = Path(output_dir)
    plots_dir = destination / "plots"
    destination.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = destination / "metrics.json"
    report = {
        "task": result.task,
        "sample_count": result.metrics["sample_count"],
        "class_names": result.class_names,
        "loss": result.loss,
        "metrics": result.metrics,
    }
    with metrics_path.open("w", encoding="utf-8") as output:
        json.dump(report, output, indent=2, ensure_ascii=False, allow_nan=False)
        output.write("\n")

    predictions_path = destination / "predictions.csv"
    _write_predictions(result, predictions_path)
    plot_paths = save_evaluation_plots(result, plots_dir, history=history)
    return {"metrics": metrics_path, "predictions": predictions_path, **plot_paths}


def evaluate_training_run(
    model: nn.Module,
    validation_loader: Iterable[tuple[Tensor, Tensor]],
    output_dir: str | Path,
    *,
    task: str,
    class_names: Sequence[str] | None = None,
    threshold: float = 0.5,
    device: str | torch.device | None = None,
    loss_function: nn.Module | None = None,
    history: Sequence[Mapping[str, float]] | None = None,
) -> EvaluationResult:
    """Load ``best_model.pth``, evaluate it, and create Phase 7 output files."""
    directory = Path(output_dir)
    checkpoint_path = directory / "best_model.pth"
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except FileNotFoundError as error:
        raise FileNotFoundError(f"Best model checkpoint does not exist: {checkpoint_path}") from error
    except (OSError, RuntimeError, ValueError) as error:
        raise ValueError(f"Could not read best model checkpoint {checkpoint_path}: {error}") from error
    if not isinstance(checkpoint, Mapping) or not isinstance(
        checkpoint.get("model_state_dict"), Mapping
    ):
        raise ValueError(f"Invalid best model checkpoint: {checkpoint_path}")
    try:
        model.load_state_dict(checkpoint["model_state_dict"])
    except RuntimeError as error:
        raise ValueError(f"Best model checkpoint does not match the supplied model: {error}") from error

    result = evaluate_model(
        model,
        validation_loader,
        task=task,
        class_names=class_names,
        threshold=threshold,
        device=device,
        loss_function=loss_function,
    )
    save_evaluation_artifacts(result, directory, history=history)
    return result
