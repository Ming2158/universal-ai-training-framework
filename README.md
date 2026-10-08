# Universal AI Training Framework

UATF 1.0.0 is a local PyTorch framework for training and evaluating image classification models. It supports single-label and multi-label classification through the same dataset, model, training, and evaluation pipeline.

## Supported Features

- Folder and CSV datasets with image validation and deterministic train/validation splits.
- ResNet50, ConvNeXt-Tiny, EfficientNet-B0, and ViT-B/16 models.
- YAML configuration, CLI overrides, CPU/CUDA selection, optional CUDA AMP, StepLR, and early stopping.
- Best/last checkpoints, resume through the Python `Trainer` API, metrics, prediction CSV, and plots.
- A local Gradio interface that uses the same CLI training pipeline.
- Per-run effective config, dataset summary, model details, metrics, checkpoint names, environment information, timestamps, and log file.

Object detection and segmentation are not implemented in this release. See [Architecture](docs/architecture.md) for the modules those tasks would extend.

## Installation

Use Python 3.10 or newer. In PowerShell, create an environment and install a PyTorch build appropriate for the machine. This example uses CUDA 12.8 wheels:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

For CPU-only use, install the PyTorch wheels appropriate for the platform instead of the CUDA command. The project dependencies and tested development tools are declared in `pyproject.toml`.

See the full [Installation guide](docs/installation.md) for CPU/GPU verification.

## Dataset Format

Single-label folder datasets have one directory per class. Use either one root folder or explicit train and validation folders:

```text
dataset/
  train/
    cats/
    dogs/
  validation/
    cats/
    dogs/
```

If there are no split folders, UATF makes a seeded random split from the class folders. Single-label CSV uses `image_path,label`. Multi-label CSV uses `image_path` and one binary column per label, with images beside the CSV or in an `images/` subdirectory:

```text
dataset/
  annotations.csv
  images/
    image-001.jpg
```

Image paths in CSV files are relative to the configured image directory. More examples and validation rules are in [Dataset Format](docs/phase4-dataset-system.md).

## Usage

The example config expects a multi-label dataset at `datasets/example/` with `annotations.csv` and `images/`:

```powershell
.\.venv\Scripts\python.exe train.py --config configs/example.yaml
```

Override common settings without editing YAML:

```powershell
.\.venv\Scripts\python.exe train.py --config configs/example.yaml --epochs 3 --batch-size 8 --device cpu
```

Each run creates a new directory under `outputs/`, containing the effective `config.yaml`, `metrics.json`, `predictions.csv`, plots, `best_model.pth`, `last_model.pth`, and `logs/train.log`. Pretrained weights are disabled by default; enabling them may download weights from torchvision.

Start the local UI from the repository root:

```powershell
.\.venv\Scripts\python.exe -m app.app
```

The UI accepts ZIP files for the two documented dataset layouts. See [Phase 9 UI](docs/phase9-gradio-ui.md).

For a small learning example, run the MNIST script. Its first run downloads the dataset if it is not cached:

```powershell
.\.venv\Scripts\python.exe scripts/train_mnist.py --epochs 1 --batch-size 128
```

## Testing and Lint

```powershell
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe scripts/gpu_smoke_test.py
```

The CUDA smoke test reports a clear failure when CUDA is unavailable. Unit tests do not download pretrained model weights. The MNIST training example may download its dataset on first use.

## Project Structure

```text
app/app.py                 Gradio UI
configs/                   Default and example YAML configs
datasets/                  Optional local datasets; data is ignored by Git
docs/                      Architecture, troubleshooting, and phase guides
outputs/                   Experiment artifacts; ignored by Git
scripts/                   MNIST example and GPU smoke test
tests/unit/                Dataset, model, evaluation, and config tests
tests/integration/         Trainer and CLI workflow tests
uatf/data/                 Image datasets, transforms, DataLoader helpers
uatf/models/               Model registry and task losses
uatf/training/             Trainer, validation, checkpoint, resume, AMP
uatf/evaluation/           Metrics, reports, predictions, and plots
uatf/config.py             YAML loading, merging, and validation
uatf/logging_utils.py      Console and experiment-file logging
uatf/cli.py                Config-driven training entry point
```

## Documentation

- [Architecture](docs/architecture.md)
- [Installation](docs/installation.md)
- [Troubleshooting](docs/troubleshooting.md)
- [Model system](docs/phase5-model-system.md)
- [Training engine](docs/phase6-training-engine.md)
- [Evaluation](docs/phase7-evaluation-system.md)
- [Configuration and CLI](docs/phase8-configuration-system.md)
- [Gradio UI](docs/phase9-gradio-ui.md)
- [Engineering Review](docs/engineering-review.md)
- [Changelog](CHANGELOG.md)
- [UATF 1.0.0 Release Notes](docs/release-notes-v1.0.0.md)

## Scope and Limitations

UATF 1.0.0 focuses on image classification on one machine. Random splitting is not stratified or group-aware. The CLI does not expose resume even though `Trainer.resume()` is available through the Python API. Exact dataset-content checksums and managed experiment storage are not provided. See [Troubleshooting](docs/troubleshooting.md) for common environment, data, and checkpoint issues.
