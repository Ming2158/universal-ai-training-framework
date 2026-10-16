"""Local image inference workflows backed by registered model assets."""

from __future__ import annotations

import csv
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from PIL import Image, UnidentifiedImageError
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, TensorDataset

from uatf.data import (
    CsvImageClassificationDataset,
    CsvMultiLabelDataset,
    ImageFolderDataset,
    build_image_transform,
    collate_image_batch,
    collate_unlabeled_image_batch,
)
from uatf.data.datasets import IMAGE_EXTENSIONS, DatasetValidationError
from uatf.evaluation.evaluator import ModelPredictions, predict_model
from uatf.experiments import load_model_asset_bundle
from uatf.models.factory import normalize_task


def asset_inference_context(output_directory: str | Path, asset_id: str) -> dict[str, Any]:
    """Load model and validate the asset's task, labels, and preprocessing contract."""
    bundle = load_model_asset_bundle(output_directory, asset_id)
    configuration = bundle["configuration"]
    model_config = configuration.get("model", {})
    dataset_config = configuration.get("dataset", {})
    task_config = configuration.get("task", {})
    task = task_config if isinstance(task_config, str) else task_config.get("type")
    task = normalize_task(task or bundle["asset"].get("task", "classification"))
    class_names = bundle["asset"].get("class_names")
    if not isinstance(class_names, list) or not class_names:
        raise ValueError(f"Model asset {asset_id} has no recoverable label names.")
    output_count = model_config.get("num_classes", model_config.get("num_labels"))
    if output_count != len(class_names):
        raise ValueError(
            f"Model asset has {output_count} outputs but {len(class_names)} saved label names."
        )
    if task == "multi_label" and dataset_config.get("format") != "csv_multilabel":
        raise ValueError("Multi-label model assets must use the csv_multilabel dataset format.")
    image_size = dataset_config.get("image_size")
    resize_enabled = dataset_config.get("resize_enabled", True)
    mean = dataset_config.get("mean")
    std = dataset_config.get("std")
    if image_size is None or mean is None or std is None:
        raise ValueError("Model asset is missing image_size, mean, or std preprocessing settings.")
    if (
        not isinstance(mean, (list, tuple))
        or not isinstance(std, (list, tuple))
        or len(mean) != 3
        or len(std) != 3
        or any(not isinstance(value, (int, float)) or not math.isfinite(value) for value in (*mean, *std))
        or any(value <= 0 for value in std)
    ):
        raise ValueError("Model asset mean and std must contain three finite RGB values; std must be positive.")
    try:
        transform = build_image_transform(
            image_size=image_size,
            mean=mean,
            std=std,
            resize_enabled=resize_enabled,
        )
    except (TypeError, ValueError) as error:
        raise ValueError(f"Model asset preprocessing configuration is invalid: {error}") from error
    return {
        **bundle,
        "task": task,
        "class_names": list(class_names),
        "transform": transform,
        "resize_enabled": resize_enabled,
        "preprocessing": {
            "image_size": image_size,
            "resize_enabled": resize_enabled,
            "mean": mean,
            "std": std,
        },
    }


def _resolve_csv_path(path: Path) -> Path:
    if path.is_dir():
        return path / "annotations.csv"
    return path


def create_evaluation_loader(
    context: dict[str, Any],
    dataset_path: str | Path,
    *,
    data_format: str | None = None,
    image_root: str | Path | None = None,
    batch_size: int = 32,
) -> tuple[DataLoader, dict[str, Any]]:
    """Create a labelled loader using the asset's task and saved label order."""
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer.")
    path = Path(dataset_path).expanduser().resolve()
    task = context["task"]
    config_format = context["configuration"].get("dataset", {}).get("format")
    selected_format = data_format or config_format
    if task == "classification":
        allowed_formats = {"image_folder", "csv_classification"}
    else:
        allowed_formats = {"csv_multilabel"}
    if selected_format not in allowed_formats:
        raise ValueError(
            f"Dataset format {selected_format!r} does not match task {task!r}; "
            f"expected one of {', '.join(sorted(allowed_formats))}."
        )

    names = context["class_names"]
    if selected_format == "image_folder":
        dataset = ImageFolderDataset(path, transform=context["transform"])
        if dataset.classes != names:
            raise DatasetValidationError(
                "Evaluation folder classes do not match the model asset's saved label order. "
                f"Expected {names}; found {dataset.classes}."
            )
    else:
        csv_path = _resolve_csv_path(path)
        if image_root is not None:
            resolved_image_root = Path(image_root).expanduser().resolve()
        elif path.is_dir() and (path / "images").is_dir():
            resolved_image_root = path / "images"
        elif csv_path.parent.joinpath("images").is_dir():
            resolved_image_root = csv_path.parent / "images"
        else:
            resolved_image_root = csv_path.parent
        if selected_format == "csv_classification":
            dataset = CsvImageClassificationDataset(
                csv_path,
                image_root=resolved_image_root,
                transform=context["transform"],
                class_names=names,
            )
        else:
            dataset = CsvMultiLabelDataset(
                csv_path,
                image_root=resolved_image_root,
                transform=context["transform"],
                label_columns=names,
            )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=None if context["resize_enabled"] else collate_image_batch,
    )
    return loader, {
        "path": str(path),
        "format": selected_format,
        "sample_count": len(dataset),
        "image_root": str(getattr(dataset, "root", getattr(dataset, "image_root", path))),
    }


class ImageDirectoryDataset(Dataset[Tensor]):
    """Read unlabeled images in stable relative-path order."""

    def __init__(self, root: str | Path, transform: Any) -> None:
        self.root = Path(root).expanduser().resolve()
        if not self.root.is_dir():
            raise DatasetValidationError(f"Image directory does not exist: {self.root}")
        paths = sorted(
            (path for path in self.root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS),
            key=lambda path: path.relative_to(self.root).as_posix().casefold(),
        )
        if not paths:
            raise DatasetValidationError(f"Image directory contains no supported images: {self.root}")
        self.samples = [(path, None) for path in paths]
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tensor:
        path = self.samples[index][0]
        try:
            with Image.open(path) as source:
                image = source.convert("RGB")
        except (OSError, SyntaxError, ValueError, UnidentifiedImageError) as error:
            raise DatasetValidationError(f"Could not read image: {path}") from error
        return self.transform(image)


def prediction_rows(result: ModelPredictions) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, (sample_id, relative_path) in enumerate(
        zip(result.sample_ids, result.relative_paths, strict=True)
    ):
        row: dict[str, Any] = {
            "sample_id": sample_id,
            "relative_path": relative_path,
        }
        if result.targets is not None:
            targets = result.targets[index]
            if result.task == "classification":
                row["target"] = result.class_names[int(targets)]
            else:
                row["target_labels"] = [
                    name
                    for name, enabled in zip(result.class_names, targets.tolist(), strict=True)
                    if enabled
                ]
        if result.task == "classification":
            prediction = int(result.predictions[index])
            row["prediction"] = result.class_names[prediction]
            row["prediction_probability"] = float(result.probabilities[index, prediction])
            row["scores"] = {
                name: float(value)
                for name, value in zip(result.class_names, result.logits[index].tolist(), strict=True)
            }
            row["probabilities"] = {
                name: float(value)
                for name, value in zip(
                    result.class_names, result.probabilities[index].tolist(), strict=True
                )
            }
        else:
            row["prediction_labels"] = [
                name
                for name, enabled in zip(
                    result.class_names, result.predictions[index].tolist(), strict=True
                )
                if enabled
            ]
            for label_index, name in enumerate(result.class_names):
                row[f"score_{name}"] = float(result.logits[index, label_index])
                row[f"probability_{name}"] = float(result.probabilities[index, label_index])
                row[f"prediction_{name}"] = int(result.predictions[index, label_index])
                if result.targets is not None:
                    row[f"target_{name}"] = int(result.targets[index, label_index])
        rows.append(row)
    return rows


def save_prediction_csv(result: ModelPredictions, path: str | Path) -> Path:
    """Write stable IDs, relative paths, scores, probabilities, and predictions."""
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    rows = prediction_rows(result)
    if result.task == "classification":
        for index, row in enumerate(rows):
            row.pop("scores", None)
            row.pop("probabilities", None)
            row.update(
                {
                    f"score_{name}": float(result.logits[index, label_index])
                    for label_index, name in enumerate(result.class_names)
                }
            )
            row.update(
                {
                    f"probability_{name}": float(result.probabilities[index, label_index])
                    for label_index, name in enumerate(result.class_names)
                }
            )
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with destination.open("x", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    return destination


def predict_model_asset_image(
    output_directory: str | Path,
    asset_id: str,
    image_path: str | Path,
    *,
    threshold: float = 0.5,
    device: str | torch.device = "auto",
) -> dict[str, Any]:
    context = asset_inference_context(output_directory, asset_id)
    path = Path(image_path).expanduser().resolve()
    if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
        raise DatasetValidationError(f"Not a supported image file: {path}")
    try:
        with Image.open(path) as source:
            tensor = context["transform"](source.convert("RGB"))
    except (OSError, SyntaxError, ValueError, UnidentifiedImageError) as error:
        raise DatasetValidationError(f"Could not read image: {path}") from error
    result = predict_model(
        context["model"],
        DataLoader(TensorDataset(tensor.unsqueeze(0)), batch_size=1),
        task=context["task"],
        class_names=context["class_names"],
        threshold=threshold,
        device=device,
    )
    scores = {
        name: float(value)
        for name, value in zip(context["class_names"], result.logits[0].tolist(), strict=True)
    }
    probabilities = {
        name: float(value)
        for name, value in zip(context["class_names"], result.probabilities[0].tolist(), strict=True)
    }
    prediction: str | list[str]
    if context["task"] == "classification":
        prediction = context["class_names"][int(result.predictions[0])]
    else:
        prediction = [
            name
            for name, enabled in zip(
                context["class_names"], result.predictions[0].tolist(), strict=True
            )
            if enabled
        ]
    return {
        "asset_id": asset_id,
        "image": str(path),
        "task": context["task"],
        "prediction": prediction,
        "decision_threshold": threshold if context["task"] == "multi_label" else None,
        "output_type": "logit scores and post-processed probabilities",
        "scores": scores,
        "probabilities": probabilities,
    }


def predict_model_asset_directory(
    output_directory: str | Path,
    asset_id: str,
    image_directory: str | Path,
    *,
    threshold: float = 0.5,
    batch_size: int = 32,
    device: str | torch.device = "auto",
    csv_path: str | Path | None = None,
) -> tuple[ModelPredictions, Path]:
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer.")
    context = asset_inference_context(output_directory, asset_id)
    dataset = ImageDirectoryDataset(image_directory, context["transform"])
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=None if context["resize_enabled"] else collate_unlabeled_image_batch,
    )
    result = predict_model(
        context["model"],
        loader,
        task=context["task"],
        class_names=context["class_names"],
        threshold=threshold,
        device=device,
    )
    if csv_path is None:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        csv_path = Path(output_directory).expanduser().resolve() / "predictions" / f"{asset_id}-{timestamp}.csv"
    return result, save_prediction_csv(result, csv_path)


__all__ = [
    "ImageDirectoryDataset",
    "asset_inference_context",
    "create_evaluation_loader",
    "predict_model_asset_directory",
    "predict_model_asset_image",
    "prediction_rows",
    "save_prediction_csv",
]
