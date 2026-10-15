"""Run whole-dataset evaluation and write experiment artifacts."""

from __future__ import annotations

import csv
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.utils.data import SequentialSampler, Subset

from uatf.evaluation.metrics import (
    _precision_recall_curve,
    _roc_curve,
    classification_metrics,
    multilabel_metrics,
)
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
    logits: Tensor | None = None
    pr_curves: dict[str, dict[str, Any]] = field(default_factory=dict)
    sample_ids: list[str] = field(default_factory=list)
    relative_paths: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    artifacts: dict[str, Path] | None = None
    metadata: dict[str, Any] | None = None


@dataclass
class ModelPredictions:
    """Raw scores and task-specific predictions for labelled or unlabelled images."""

    task: str
    class_names: list[str]
    logits: Tensor
    predictions: Tensor
    probabilities: Tensor
    targets: Tensor | None
    sample_ids: list[str]
    relative_paths: list[str]
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


def _sample_metadata(loader: Iterable[Any], sample_count: int) -> tuple[list[str], list[str]]:
    dataset = getattr(loader, "dataset", None)
    sampler = getattr(loader, "sampler", None)
    base_dataset = dataset
    indices = list(range(len(dataset))) if dataset is not None else []
    while isinstance(base_dataset, Subset):
        indices = [base_dataset.indices[index] for index in indices]
        base_dataset = base_dataset.dataset

    samples = getattr(base_dataset, "samples", None)
    root = getattr(base_dataset, "root", getattr(base_dataset, "image_root", None))
    if (
        samples is None
        or root is None
        or len(indices) != sample_count
        or not isinstance(sampler, SequentialSampler)
    ):
        ids = [f"sample-{index:06d}" for index in range(sample_count)]
        return ids, [""] * sample_count

    root_path = Path(root).resolve()
    ids: list[str] = []
    relative_paths: list[str] = []
    for index in indices:
        sample_path = Path(samples[index][0]).resolve()
        try:
            relative = sample_path.relative_to(root_path).as_posix()
        except ValueError:
            relative = sample_path.name
        ids.append(relative)
        relative_paths.append(relative)
    return ids, relative_paths


def predict_model(
    model: nn.Module,
    data_loader: Iterable[Any],
    *,
    task: str,
    class_names: Sequence[str] | None = None,
    threshold: float = 0.5,
    device: str | torch.device | None = None,
    loss_function: nn.Module | None = None,
    require_targets: bool = False,
) -> ModelPredictions:
    """Run shared, ordered inference over labelled or unlabelled batches."""
    normalized_task = normalize_task(task)
    names = list(class_names) if class_names is not None else _infer_class_names(data_loader)
    resolved_device = _model_device(model, device)
    all_targets: list[Tensor] = []
    all_logits: list[Tensor] = []
    weighted_loss = 0.0
    sample_count = 0
    saw_targets: bool | None = None

    model.to(resolved_device)
    model.eval()
    with torch.no_grad():
        for batch_index, batch in enumerate(data_loader, start=1):
            if isinstance(batch, Tensor):
                inputs, targets = batch, None
            elif isinstance(batch, (tuple, list)) and len(batch) in {1, 2}:
                inputs = batch[0]
                targets = batch[1] if len(batch) == 2 else None
            else:
                raise ValueError(
                    f"Inference batch {batch_index} must be inputs or (inputs, targets)."
                )
            if not isinstance(inputs, Tensor) or inputs.ndim == 0:
                raise ValueError(f"Inference batch {batch_index} must contain batched input tensors.")
            if targets is not None:
                if not isinstance(targets, Tensor) or targets.ndim == 0:
                    raise ValueError(f"Inference batch {batch_index} has invalid targets.")
                if inputs.shape[0] != targets.shape[0]:
                    raise ValueError(f"Inference batch {batch_index} has incompatible batch shapes.")
            if saw_targets is not None and (targets is not None) != saw_targets:
                raise ValueError("Inference batches cannot mix labelled and unlabelled samples.")
            saw_targets = targets is not None
            if require_targets and targets is None:
                raise ValueError("Evaluation requires ground-truth labels.")

            logits = model(inputs.to(resolved_device))
            if not isinstance(logits, Tensor) or logits.ndim != 2:
                raise ValueError("Models must return a two-dimensional (N x C) tensor.")
            if logits.shape[0] != inputs.shape[0]:
                raise ValueError(f"Inference batch {batch_index} has incompatible output size.")
            if not torch.isfinite(logits).all().item():
                raise FloatingPointError(f"Model output contains NaN or infinity in batch {batch_index}.")
            if names is not None and len(names) != logits.shape[1]:
                raise ValueError(
                    f"Model returns {logits.shape[1]} outputs, but the asset has {len(names)} labels."
                )

            if targets is not None:
                targets = targets.to(resolved_device)
                if normalized_task == "classification":
                    if targets.ndim != 1:
                        raise ValueError("Classification targets must be one-dimensional class IDs.")
                elif targets.ndim != 2 or targets.shape != logits.shape:
                    raise ValueError("Multi-label targets must match model outputs with shape (N x C).")
                if loss_function is not None:
                    loss = loss_function(logits, targets)
                    if loss.numel() != 1 or not torch.isfinite(loss).all().item():
                        raise FloatingPointError(f"Evaluation loss is invalid in batch {batch_index}.")
                    scale = 1 if getattr(loss_function, "reduction", "mean") == "sum" else targets.shape[0]
                    weighted_loss += float(loss.item()) * scale
                all_targets.append(targets.detach().cpu())
            all_logits.append(logits.detach().cpu())
            sample_count += inputs.shape[0]

    if sample_count == 0:
        raise ValueError("Inference DataLoader yielded no examples.")
    logits_values = torch.cat(all_logits, dim=0)
    target_values = torch.cat(all_targets, dim=0) if all_targets else None
    resolved_names = names or [f"class_{index}" for index in range(logits_values.shape[1])]
    if normalized_task == "classification":
        probabilities = torch.softmax(logits_values, dim=1)
        predictions = probabilities.argmax(dim=1)
    else:
        if not 0 <= threshold <= 1:
            raise ValueError("threshold must be between 0 and 1 for multi-label inference.")
        probabilities = torch.sigmoid(logits_values)
        predictions = (probabilities >= threshold).to(torch.int64)
    sample_ids, relative_paths = _sample_metadata(data_loader, sample_count)
    return ModelPredictions(
        task=normalized_task,
        class_names=resolved_names,
        logits=logits_values,
        predictions=predictions,
        probabilities=probabilities,
        targets=target_values,
        sample_ids=sample_ids,
        relative_paths=relative_paths,
        loss=weighted_loss / sample_count if loss_function is not None else None,
    )


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
    output = predict_model(
        model,
        data_loader,
        task=task,
        class_names=class_names,
        threshold=threshold,
        device=device,
        loss_function=loss_function,
        require_targets=True,
    )
    assert output.targets is not None
    if output.task == "classification":
        metrics = classification_metrics(
            output.targets,
            output.predictions,
            num_classes=output.logits.shape[1],
            class_names=output.class_names,
        )
        roc_curves = {}
        pr_curves = {}
        valid_auc: list[float] = []
        valid_ap: list[float] = []
        warnings: list[str] = []
        for index, name in enumerate(output.class_names):
            binary_targets = output.targets == index
            roc = _roc_curve(binary_targets, output.probabilities[:, index])
            pr = _precision_recall_curve(binary_targets, output.probabilities[:, index])
            roc_curves[name] = roc
            pr_curves[name] = pr
            metrics["per_class"][name]["auc"] = roc["auc"]
            metrics["per_class"][name]["average_precision"] = pr["average_precision"]
            if roc["auc"] is not None:
                valid_auc.append(roc["auc"])
            else:
                warnings.append(f"{name}: ROC-AUC requires positive and negative samples.")
            if pr["average_precision"] is not None:
                valid_ap.append(pr["average_precision"])
            else:
                warnings.append(f"{name}: average precision requires at least one positive sample.")
        metrics["macro_auc"] = sum(valid_auc) / len(valid_auc) if valid_auc else None
        metrics["macro_average_precision"] = sum(valid_ap) / len(valid_ap) if valid_ap else None
        resolved_names = output.class_names
    else:
        metrics = multilabel_metrics(
            output.targets,
            output.probabilities,
            class_names=output.class_names,
            threshold=threshold,
        )
        roc_curves = metrics.pop("roc_curves")
        metrics.pop("micro_roc_curve")
        pr_curves = {}
        warnings = []
        for index, name in enumerate(output.class_names):
            curve = _precision_recall_curve(output.targets[:, index].bool(), output.probabilities[:, index])
            pr_curves[name] = curve
            metrics["per_class"][name]["average_precision"] = curve["average_precision"]
            if roc_curves[name]["auc"] is None:
                warnings.append(f"{name}: ROC-AUC requires positive and negative samples.")
            if curve["average_precision"] is None:
                warnings.append(f"{name}: average precision requires at least one positive sample.")
        resolved_names = output.class_names

    return EvaluationResult(
        task=output.task,
        class_names=resolved_names,
        targets=output.targets,
        logits=output.logits,
        predictions=output.predictions,
        probabilities=output.probabilities,
        metrics=metrics,
        roc_curves=roc_curves,
        pr_curves=pr_curves,
        sample_ids=output.sample_ids,
        relative_paths=output.relative_paths,
        warnings=warnings,
        loss=output.loss,
    )


def _write_predictions(result: EvaluationResult, path: Path) -> None:
    if result.targets is None:
        raise ValueError("Evaluation predictions require ground-truth targets.")
    target_rows = result.targets.tolist()
    prediction_rows = result.predictions.tolist()
    probability_rows = result.probabilities.tolist()
    sample_ids = result.sample_ids or [f"sample-{index:06d}" for index in range(len(target_rows))]
    relative_paths = result.relative_paths or [""] * len(target_rows)
    if result.task == "classification":
        columns = ["sample_index", "sample_id", "relative_path", "target", "prediction", "confidence"]
        if result.logits is not None:
            columns.extend(f"score_{name}" for name in result.class_names)
        columns.extend(f"probability_{name}" for name in result.class_names)
        with path.open("w", encoding="utf-8", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=columns)
            writer.writeheader()
            for index, (target, prediction, probabilities, sample_id, relative_path) in enumerate(
                zip(
                    target_rows,
                    prediction_rows,
                    probability_rows,
                    sample_ids,
                    relative_paths,
                    strict=True,
                )
            ):
                row = {
                    "sample_index": index,
                    "sample_id": sample_id,
                    "relative_path": relative_path,
                    "target": result.class_names[int(target)],
                    "prediction": result.class_names[int(prediction)],
                    "confidence": max(probabilities),
                }
                if result.logits is not None:
                    row.update(
                        {
                            f"score_{name}": value
                            for name, value in zip(
                                result.class_names, result.logits[index].tolist(), strict=True
                            )
                        }
                    )
                row.update(
                    {
                        f"probability_{name}": probability
                        for name, probability in zip(result.class_names, probabilities, strict=True)
                    }
                )
                writer.writerow(row)
        return

    columns = ["sample_index", "sample_id", "relative_path"]
    columns.extend(f"target_{name}" for name in result.class_names)
    if result.logits is not None:
        columns.extend(f"score_{name}" for name in result.class_names)
    columns.extend(f"prediction_{name}" for name in result.class_names)
    columns.extend(f"probability_{name}" for name in result.class_names)
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=columns)
        writer.writeheader()
        for index, (targets, predictions, probabilities, sample_id, relative_path) in enumerate(
            zip(
                target_rows,
                prediction_rows,
                probability_rows,
                sample_ids,
                relative_paths,
                strict=True,
            )
        ):
            row: dict[str, Any] = {
                "sample_index": index,
                "sample_id": sample_id,
                "relative_path": relative_path,
            }
            row.update(
                {f"target_{name}": int(value) for name, value in zip(result.class_names, targets, strict=True)}
            )
            if result.logits is not None:
                row.update(
                    {
                        f"score_{name}": value
                        for name, value in zip(
                            result.class_names, result.logits[index].tolist(), strict=True
                        )
                    }
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
    save_predictions: bool = True,
    save_plots: bool = True,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Path]:
    """Write metrics, predictions, and plots under one experiment directory."""
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    metrics_path = destination / "metrics.json"
    report = {
        "task": result.task,
        "sample_count": result.metrics["sample_count"],
        "class_names": result.class_names,
        "loss": result.loss,
        "metrics": result.metrics,
        "warnings": result.warnings,
    }
    if metadata is not None:
        report["metadata"] = dict(metadata)
    with metrics_path.open("w", encoding="utf-8") as output:
        json.dump(report, output, indent=2, ensure_ascii=False, allow_nan=False)
        output.write("\n")

    artifacts = {"metrics": metrics_path}
    if save_predictions:
        predictions_path = destination / "predictions.csv"
        _write_predictions(result, predictions_path)
        artifacts["predictions"] = predictions_path
    if save_plots:
        plots_dir = destination / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)
        artifacts.update(save_evaluation_plots(result, plots_dir, history=history))
    return artifacts


def evaluate_model_asset(
    output_directory: str | Path,
    asset_id: str,
    dataset_path: str | Path,
    *,
    data_format: str | None = None,
    image_root: str | Path | None = None,
    result_directory: str | Path | None = None,
    batch_size: int = 32,
    threshold: float | None = None,
    device: str | torch.device = "auto",
) -> EvaluationResult:
    """Evaluate a registered model asset into a new, isolated result directory."""
    from uatf.inference import asset_inference_context, create_evaluation_loader
    from uatf.models.losses import get_loss

    root = Path(output_directory).expanduser().resolve()
    context = asset_inference_context(root, asset_id)
    effective_threshold = (
        float(context["configuration"].get("evaluation", {}).get("threshold", 0.5))
        if threshold is None
        else threshold
    )
    loader, dataset_metadata = create_evaluation_loader(
        context,
        dataset_path,
        data_format=data_format,
        image_root=image_root,
        batch_size=batch_size,
    )
    result = evaluate_model(
        context["model"],
        loader,
        task=context["task"],
        class_names=context["class_names"],
        threshold=effective_threshold,
        device=device,
        loss_function=get_loss(context["task"]),
    )

    if result_directory is None:
        destination_root = root / "evaluations"
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        destination = destination_root / f"{asset_id}-{timestamp}"
    else:
        destination = Path(result_directory).expanduser().resolve()
    try:
        relative_destination = destination.relative_to(root)
    except ValueError:
        relative_destination = None
    if relative_destination is not None and (
        not relative_destination.parts or relative_destination.parts[0] != "evaluations"
    ):
        raise ValueError("Evaluation output inside the model store must be under its evaluations directory.")
    for parent in destination.parents:
        if parent == root:
            break
        if (parent / "experiment.json").is_file():
            raise ValueError("Independent evaluation output cannot be placed inside a training run.")
    if destination.exists():
        raise FileExistsError(f"Evaluation result directory already exists: {destination}")

    metadata = {
        "asset_id": asset_id,
        "experiment_id": context["asset"].get("experiment_id"),
        "checkpoint_identifier": f"{asset_id}:{Path(context['asset']['weights_path']).name}",
        "checkpoint": context["asset"].get("weights_path"),
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": dataset_metadata,
        "configuration": {
            "task": context["task"],
            "data_format": dataset_metadata["format"],
            "batch_size": batch_size,
            "threshold": effective_threshold if context["task"] == "multi_label" else None,
            "device_requested": str(device),
            "preprocessing": context["preprocessing"],
        },
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.mkdir(exist_ok=False)
    result.artifacts = save_evaluation_artifacts(result, destination, metadata=metadata)
    result.metadata = metadata
    return result


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
    save_predictions: bool = True,
    save_plots: bool = True,
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
    save_evaluation_artifacts(
        result,
        directory,
        history=history,
        save_predictions=save_predictions,
        save_plots=save_plots,
    )
    return result
