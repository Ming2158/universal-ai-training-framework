# UATF V1.0.0 to V2.0.0 Migration

V2.0.0 extends the local image-classification workflow with dataset inspection, task records, experiment and model-asset indexes, independent evaluation, and inference. It retains the V1 configuration schema and existing experiment directories.

V2.0.0 is the current release. Package metadata is `2.0.0`; the existing `v1.0.0` tag remains unchanged. See the [release notes](release-notes-v2.0.0.md) and [release acceptance record](development/v2.0.0-release-acceptance.md) for verified behavior and known gaps.

## Before upgrading

Keep the existing environment and output directory until a new environment has passed its own installation checks. The framework does not migrate, move, or delete existing datasets, checkpoints, or experiment directories.

Create a separate Python 3.10+ environment and install a PyTorch/torchvision pair that matches the machine. For the CUDA 12.8 development environment:

```powershell
py -3.12 -m venv .venv-v2
.\.venv-v2\Scripts\python.exe -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
.\.venv-v2\Scripts\python.exe -m pip install -e ".[dev]"
```

For CPU-only use, install the platform's CPU PyTorch wheels before installing UATF. See [Installation](installation.md) for environment checks.

## YAML configurations

Existing V1 YAML files remain accepted by the current configuration resolver. No bulk YAML rewrite is required. Continue to run a config with the same entry point:

```powershell
.\.venv-v2\Scripts\python.exe train.py --config configs/example.yaml
```

The current schema adds configuration for optimizer and scheduler choices, augmentation, task-compatible loss, and checkpoint resume. Unknown keys and invalid task/data combinations are rejected with a field-specific error. Review `configs/default.yaml` and [Configuration](configuration.md) before adding those fields. Generated effective `config.yaml` files include normalized values and are intended for this or a newer UATF version.

CLI overrides remain available for epochs, batch size, learning rate, seed, device, and output directory. The interactive CLI wizard is additive. There is no dedicated `--resume` flag; set `training.resume_from` in YAML or use the Python API.

## Checkpoints and experiment directories

V1 experiment folders remain readable as legacy records and, when their configuration and best checkpoint are recoverable, as read-only model assets. V2 does not rewrite those folders to add metadata. New runs create experiment records and references to dataset split manifests and model assets beneath the configured output root.

Resume from a compatible full training checkpoint:

```yaml
training:
  resume_from: outputs/experiment_001/last_model.pth
```

`last_model.pth` contains optimizer and training state and is the normal resume input. Exported inference weights such as `.weights.pth` contain model state only and cannot restore optimizer, scheduler, or epoch history. Resume validates the task, labels, model, preprocessing, split, and available optimizer state; incompatible inputs fail with an error rather than silently starting from a different state.

Keep the output root stable when opening historical runs. The experiment browser reads older layouts without moving them. The V2 `.uatf` registry and exports are additional local files; they do not replace source checkpoints.

## Python API

The `uatf.api` facade is additive and calls the shared configuration, experiment, task, training, evaluation, and inference modules:

```python
from uatf.api import run_training

run_directory = run_training(
    "configs/example.yaml",
    overrides={"training": {"epochs": 2}},
)
```

Existing lower-level imports such as `Trainer`, datasets, model builders, and evaluation helpers remain available. Model assets can be evaluated or used for single-image and directory predictions through the API described in [Independent Evaluation and Inference](phase6-independent-evaluation.md).

## GUI changes

The existing local Gradio entry point remains `python -m app.app`. V2 adds dataset inspection and preview, live task status and stop requests, experiment history, model assets, independent evaluation, single-image prediction, and batch CSV export. GUI training still runs through the same CLI and training engine.

The GUI serves local files and paths from the UATF host. Local dataset and inference paths must be readable by the process running Gradio. For a feature-by-feature status, see the [Phase 8 acceptance report](development/v2.0.0-phase8-acceptance.md); browser-level interaction was not part of the automated test suite.

## Compatibility summary

| Area | V1 to V2 behavior |
| --- | --- |
| YAML | Existing configurations remain accepted; new options are additive. |
| CLI | Config-driven commands and supported overrides remain; the bilingual wizard is additive. |
| Python | Existing component APIs remain; `uatf.api` adds a higher-level facade. |
| Checkpoints | Compatible full training checkpoints can resume; inference-only exports cannot. |
| Experiments | Existing directories are read as legacy records without migration writes. |
| Models and data | Existing built-in image-classification architectures and folder/CSV layouts remain supported. |

No configuration or artifact migration command is required by the current code. Existing user data, checkpoints, and experiment directories should remain in place during an upgrade.
