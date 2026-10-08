# UATF 1.0.0 Release Notes

UATF 1.0.0 provides a local training workflow for image classification. A run starts from a YAML config or the Gradio UI, validates a dataset, creates a torchvision model, trains and validates it, evaluates the best checkpoint, and saves the run artifacts in one directory.

## Included

- Single-label classification from class folders or `image_path,label` CSV.
- Multi-label classification from `image_path` plus binary label columns.
- ResNet50, ConvNeXt-Tiny, EfficientNet-B0, and ViT-B/16 with optional pretrained weights.
- YAML config merging and validation, CLI overrides, deterministic random splits, CUDA/CPU selection, and optional CUDA AMP.
- Trainer validation, StepLR, early stopping, atomic best/last checkpoint writes, and Python API resume.
- Classification and multi-label metrics, prediction CSV, plots, experiment metadata, and timestamped logs.
- A Gradio UI for ZIP uploads using the shared CLI workflow.

## Install and Run

Install Python 3.10+, a platform-compatible PyTorch/torchvision build, then the project and development dependencies:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe train.py --config configs/example.yaml
```

The example config expects `datasets/example/annotations.csv` and `datasets/example/images/`. Start the UI with `.\.venv\Scripts\python.exe -m app.app`. See [Installation](installation.md) for setup and the [README](../README.md) for folder datasets, CPU use, tests, and smoke checks.

## Experiment Record

Each run stores the effective config, dataset source and summary, model and task, Python/PyTorch/device versions, training history, final metrics, checkpoint names, UTC timestamps, predictions, plots, and `logs/train.log`. Dataset files and model weights remain local and are not part of the source release.

## Known Limits

- Only single-label and multi-label image classification are supported. Object detection and segmentation are future work.
- Dataset splitting is seeded but not stratified or group-aware.
- The CLI does not expose resume; the Python `Trainer.resume()` API does.
- The experiment record summarizes dataset path, format, sample count, and labels, but does not hash every dataset file.
- Training is local and single-device. No managed experiment service or distributed training is included.
