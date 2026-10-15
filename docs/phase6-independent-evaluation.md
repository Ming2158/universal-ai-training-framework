# Phase 6: Independent Evaluation and Inference

Phase 6 uses registered model assets and the same model factory, image transforms, dataset adapters, and evaluation metrics used by training.

## Supported Tasks and Inputs

- Single-label image classification accepts `image_folder` datasets and `csv_classification` datasets with `image_path,label` columns.
- Multi-label image classification accepts `csv_multilabel` datasets with `image_path` and binary label columns.
- Evaluation label names and order must match the model asset. Folder class directories are checked against that saved order. CSV evaluation retains labels that have no examples in the evaluation split and reports undefined ROC-AUC or average precision as `null` with a warning.
- Single-image and directory inference use the model asset's saved resize and RGB normalization settings. Single-label outputs use softmax probabilities and argmax; multi-label outputs use sigmoid probabilities and a configurable threshold.

## Python API

```python
from uatf.api import (
    evaluate_model_asset,
    predict_model_asset_directory,
    predict_model_asset_image,
)

result = evaluate_model_asset(
    "outputs/ui_runs",
    "model-<experiment-id>",
    "datasets/holdout",
    data_format="image_folder",
)
single = predict_model_asset_image(
    "outputs/ui_runs", "model-<experiment-id>", "images/example.jpg"
)
batch, csv_path = predict_model_asset_directory(
    "outputs/ui_runs", "model-<experiment-id>", "images/to-predict"
)
```

Independent evaluation creates a new directory under `outputs/ui_runs/evaluations/` by default. It writes `metrics.json`, `predictions.csv`, and task-appropriate plots. The record includes the model asset and checkpoint identifiers, evaluation dataset, timestamp, preprocessing, threshold, and requested batch/device settings.

Batch inference writes a new CSV under the model store's `predictions/` directory by default. Rows contain a stable sample ID, original relative path, prediction, raw model scores, and post-processed probabilities. Inference CSV creation refuses to overwrite an existing file.

## Exported Model Format

`export_model_asset` writes a PyTorch state-dict `.pth` file and a companion `.pth.json` sidecar. The sidecar includes the task, architecture/output count, label names when recoverable, and preprocessing configuration. `load_exported_model_asset` reconstructs the registered architecture and strictly loads the weights from that pair. This is the verified export format for Phase 6; ONNX export is not included.
