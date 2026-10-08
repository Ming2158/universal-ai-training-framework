# Installation and Troubleshooting

## Installation

Follow the [Installation guide](installation.md) to create an environment and install a platform-compatible PyTorch build.

## CUDA Is Unavailable

Check the installed Python, PyTorch build, CUDA runtime, and device:

```powershell
.\.venv\Scripts\python.exe scripts/gpu_smoke_test.py
```

Use `--device cpu` for a CPU run, or install a CUDA-enabled PyTorch build compatible with the NVIDIA driver. The CLI's `auto` device uses CUDA when PyTorch reports it available and otherwise logs a warning before using CPU.

## CUDA Out of Memory

Reduce `training.batch_size` or pass `--batch-size`. A smaller model or optional `training.amp.enabled: true` can also reduce memory use on a supported CUDA device. AMP is not enabled by default. Do not increase `num_workers` or batch size without checking memory use on the target dataset and model.

## Dataset Validation Errors

- Folder classification needs one directory per class and at least one readable image in total.
- Train and validation folders must contain the same class names in the same sorted order.
- Single-label CSV requires `image_path,label`.
- Multi-label CSV requires `image_path` and one or more label columns whose values are `0` or `1`.
- CSV image paths are resolved relative to the configured image directory. Missing or damaged images, duplicate paths, blank labels, and malformed rows stop the run with the CSV row or image path in the error.
- Random splits are not stratified or group-aware. Keep related images together in a pre-split dataset when sample-level random splitting would leak information.

## Configuration Errors

The CLI validates configuration before allocating an experiment. Check the reported field name and compare the file with `configs/default.yaml` or `configs/example.yaml`. Unknown keys are rejected so misspelled options do not silently take effect. Paths in YAML are resolved from the current working directory.

## Pretrained Weights

Pretrained weights are off by default. If enabled, torchvision may download them on the first run. Provide network access or a valid local torchvision cache; otherwise disable `model.pretrained`.

## Checkpoint Errors

Checkpoint loading requires the same model architecture and output size used when saving. `Trainer.load_checkpoint()` reports missing files, malformed state, or a model mismatch with the checkpoint path. Training writes checkpoints via a temporary file and atomic replacement; check free space and write access to the output directory if saving fails.

## Find Experiment Logs

Each CLI run creates an isolated directory under `output.directory`. Inspect `logs/train.log` for timestamped levels and a traceback, `config.yaml` for the effective options, and `metrics.json` for dataset/model metadata and results. `best_model.pth` and `last_model.pth` are the saved checkpoints.

## Recreate a Stale Virtual Environment

Moving a project directory or uninstalling its base Python can leave a virtual environment pointing at a Python executable that no longer exists. Confirm a replacement environment works before removing the old one:

```powershell
py -3.12 -m venv .venv-new
.\.venv-new\Scripts\python.exe -m pip install --upgrade pip
.\.venv-new\Scripts\python.exe -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
.\.venv-new\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv-new\Scripts\python.exe -m pytest
```

After verifying the new environment, switch to it or rename the old directory. Virtual environments are local and are not included in releases.
