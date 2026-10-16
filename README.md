<p align="center"><img src="UATF_logo.png" alt="UATF logo" width="360"></p>

# Universal AI Training Framework

[English](README.md) | [简体中文](README_zh.md)

For first-time setup, see the [UATF usage guide](使用文档.md) for the GUI and CLI workflows.

UATF 2.1.0 is a local PyTorch framework for single-label and multi-label image classification, with shared GUI, CLI, YAML, and Python API workflows.

V2.0.0 retains V1.0.0 YAML configurations and historical experiment directories while adding dataset inspection, persistent tasks, experiment and model-asset management, independent evaluation, and inference.

V2.1.0 adds optional on-the-fly resizing, variable-size batching for convolutional models, parallel data loading, and large ZIP space checks.

## Supported Features

- Folder and CSV datasets with image validation and deterministic train/validation splits.
- ResNet50, ConvNeXt-Tiny, EfficientNet-B0, and ViT-B/16 models.
- YAML configuration, CLI overrides, CPU/CUDA selection, optional CUDA AMP, Adam/AdamW/SGD, StepLR/cosine scheduling, clipping, accumulation, and early stopping.
- Default-off training augmentation with deterministic validation and inference preprocessing.
- Optional fixed-size resizing; disabling it keeps original dimensions and pads each batch for convolutional models.
- Configurable parallel image loading with persistent workers and pinned memory for CUDA training.
- Best/last checkpoints, resume through the Python `Trainer` API, metrics, prediction CSV, and plots.
- A local Gradio interface that uses the same CLI training pipeline.
- Independent evaluation of saved model assets, single-image inference, directory prediction, error review, and reloadable PyTorch model exports.
- Per-run effective config, dataset summary, model details, metrics, checkpoint names, environment information, timestamps, and log file.

See the [Phase 6 guide](docs/phase6-independent-evaluation.md) for supported evaluation formats, Python APIs, output records, and exported model sidecars.

See the [Phase 7 model and training guide](docs/phase7-model-training.md) for adapter capabilities, pretrained weight handling, augmentation settings, supported strategies, and split decisions.

Object detection, segmentation, distributed training, and managed experiment storage are outside this release. See [Architecture](docs/architecture.md) for the modules those tasks would extend.

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

On Windows, double-click `启动UATF.bat` in the repository root to start the local UI. It uses an existing `.venv` with CUDA-enabled PyTorch when available; otherwise it creates or uses a `venv` environment with CPU dependencies. Python 3.10, 3.11, or 3.12 and an internet connection are required for setup. The browser opens automatically. Close the launcher window to stop UATF.

The example config expects a multi-label dataset at `datasets/example/` with `annotations.csv` and `images/`:

```powershell
.\.venv\Scripts\python.exe train.py --config configs/example.yaml
```

Override common settings without editing YAML:

```powershell
.\.venv\Scripts\python.exe train.py --config configs/example.yaml --epochs 3 --batch-size 8 --device cpu
```

For a guided terminal workflow, run:

```powershell
.\.venv\Scripts\python.exe train.py --interactive
```

After installing the project, `uatf --interactive` starts the same wizard; running `uatf` or `python train.py` in a terminal also opens it. If the project was already installed, run `python -m pip install -e .` once to create the new `uatf` command. The wizard supports Simplified Chinese and English; type `:lang` at any prompt to switch. It guides you through the task and dataset format, a local dataset folder, CSV file, or ZIP archive, model and pretrained weights, epochs, batch size, learning rate, and CPU/CUDA device. A settings summary is shown before training begins. ZIP uploads are extracted under `outputs/cli_uploads/`; wizard runs are saved under `outputs/cli_runs/`.

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
启动UATF.bat               Windows double-click launcher
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
uatf/cli_wizard.py         Bilingual interactive training wizard
```

## Documentation

- [Architecture](docs/architecture.md)
- [Installation](docs/installation.md)
- [Troubleshooting](docs/troubleshooting.md)
- [Model system](docs/phase5-model-system.md)
- [Training engine](docs/phase6-training-engine.md)
- [Evaluation](docs/phase7-evaluation-system.md)
- [Configuration and CLI](docs/phase8-configuration-system.md)
- [Unified configuration and Python API](docs/configuration.md)
- [V1.0.0 to V2.0.0 migration guide](docs/migration-v1-to-v2.md)
- [V2.0.0 acceptance report](docs/development/v2.0.0-phase8-acceptance.md)
- [V2.0.0 release notes](docs/release-notes-v2.0.0.md)
- [V2.0.0 release acceptance](docs/development/v2.0.0-release-acceptance.md)
- [V2.1.0 release notes](docs/release-notes-v2.1.0.md)
- [Gradio UI](docs/phase9-gradio-ui.md)
- [Historical Engineering Review](docs/engineering-review.md)
- [Changelog](CHANGELOG.md)
- [Versioning and Release Rules](docs/versioning.md)
- [UATF 1.0.0 Release Notes](docs/release-notes-v1.0.0.md)

## Scope and Limitations

UATF 2.1.0 focuses on single-label and multi-label image classification on one machine. Random splitting is not stratified or group-aware. There is no dedicated CLI `--resume` flag; YAML `training.resume_from`, `run_training()`, and `Trainer.resume()` support checkpoint restoration. Exact dataset-content checksums and managed experiment storage are not provided. See [Troubleshooting](docs/troubleshooting.md) for common environment, data, and checkpoint issues.
