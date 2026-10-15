# Configuration

UATF accepts one mapping schema through YAML, the CLI wizard, the Gradio UI, and
the Python API. `load_config()` reads user values without applying defaults;
`resolve_config()` applies defaults and validation and raises
`ConfigurationError` (a `ValueError` subclass) for invalid values.

## Source precedence

The effective configuration follows this order, from lowest to highest:

1. Framework defaults in `uatf.config.DEFAULT_CONFIG`.
2. Values from the supplied YAML file or Python mapping.
3. Explicit entry-point overrides. CLI flags override matching YAML fields;
   Python callers can pass the same kind of mapping through `overrides`.

The GUI turns its form values into the same configuration mapping and passes it
to the CLI. It does not apply a second set of hidden defaults. Nested mappings
are merged, so an override to `training.epochs` keeps the other training values.

```python
from uatf.api import run_training

run_directory = run_training(
    "configs/example.yaml",
    overrides={"training": {"epochs": 5}},
)
```

`run_training()` also accepts a Python mapping and an `event_callback` that
receives `TrainingEvent` values. Events include `training_started`,
throttled `training_progress` updates with the current phase and batch,
`epoch_completed`, `training_completed`, and `training_failed`. Events carry a
UTC timestamp when delivered. Observer failures are logged and disable further
notifications without stopping training. The CLI can write events as JSON Lines
with `--events-file`; the Gradio UI polls that file for live progress.

## Supported options

The validator rejects unknown sections and keys. Current model settings are
`model.name`, `model.pretrained`, and the optional output count
`model.num_classes` (`model.num_labels` remains an alias). The built-in model
registry does not expose arbitrary constructor keyword arguments, so such
values are not accepted as model parameters.

Training supports Adam (`training.optimizer.name: adam`) and SGD
(`training.optimizer.name: sgd`). Adam accepts `weight_decay`, `betas`, and
`eps`; SGD accepts `weight_decay`, `momentum`, and `nesterov`. StepLR remains the
only scheduler. Resume uses `training.resume_from` with an existing checkpoint
file. AMP requires a CUDA device; `device: auto` is resolved at runtime and can
still fail clearly if CUDA is unavailable.

`dataset` supports image-folder, single-label CSV, and multi-label CSV formats,
validation split fraction, image size, RGB normalization, and label columns for
multi-label CSV data. DataLoader batch size and worker count remain in
`training.batch_size` and `training.num_workers` for compatibility.

Evaluation supports task-appropriate primary metrics and the multi-label
threshold. If omitted, the primary metric defaults to `accuracy` for
single-label classification and `micro_f1` for multi-label classification.
`output.save_predictions` and `output.save_plots` control optional artifacts;
metrics, logs, and best/last checkpoints continue to be written for every run.
Output paths may be created during a run, but their nearest existing parent
must be a writable directory.

## Reproducibility files

Every run writes:

- `source_config.yaml`: the supplied user mapping before defaults and overrides.
- `config.yaml`: the normalized effective configuration, including the class
  count inferred from the dataset.
- `metrics.json`: environment information, CLI/Python overrides, completed
  epoch history, and evaluation results.

Epoch learning rates and metrics belong to the run history; they do not modify
either configuration file. Existing UATF 1.0 YAML files remain accepted by the
new resolver; generated effective configs include the expanded schema and are
intended for this or a newer UATF version.
