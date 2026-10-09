# Phase 8: Configuration System

Phase 8 adds a YAML-driven training entry point. Training settings, dataset paths, model selection, evaluation options, and output location live in configuration files rather than in Python source.

## Run training

From the repository root:

```powershell
python train.py --config configs/example.yaml
```

The example expects this multi-label dataset layout:

```text
datasets/example/
  annotations.csv
  images/
    image-001.jpg
```

`annotations.csv` must have an `image_path` column and one binary `0`/`1` column per label. Image paths in the CSV are relative to `images/` by default; YAML paths are resolved from the current working directory. The dataset path can instead point directly to a CSV file; `image_dir` and `label_columns` can be set when the image folder or selected label columns differ from the defaults.

For single-label classification, point `dataset.path` to a folder with one subfolder per class. The CLI uses `train/` and `validation/` subfolders when both exist. Otherwise it splits the class folders according to `validation_fraction`. Explicit `train_dir` and `validation_dir` are also supported. A CSV single-label dataset can use `format: csv_classification` with `annotations_csv`.

Override common training values from the command line:

```powershell
python train.py --config configs/example.yaml --epochs 3 --batch-size 8 --device cpu
```

Values follow three levels: built-in defaults, YAML values, then provided CLI arguments. Current CLI overrides are `--epochs`, `--batch-size`, `--learning-rate`, `--seed`, `--device`, and `--output-dir`. `training.monitor` chooses the value used to save the best checkpoint; `evaluation.primary_metric` names the main metric in the final report.

## Interactive terminal wizard

You can create a run without writing YAML first:

```powershell
python train.py --interactive
```

The wizard supports Simplified Chinese and English. Type `:lang` at a prompt to switch languages. It asks for the task, dataset layout and local path (directory, CSV file, or ZIP archive), model, optional pretrained weights, epochs, batch size, learning rate, and device. It shows the resolved settings and asks before starting. ZIP files are safely extracted under `outputs/cli_uploads/`; the generated YAML is saved under `outputs/cli_configs/`, and experiments go to `outputs/cli_runs/`. Running `python train.py` with no arguments in a terminal also opens the wizard. An installed console script is available as `uatf --interactive`; run `python -m pip install -e .` after updating an existing installation to create it.

## Validation and outputs

The config loader uses PyYAML's safe loader, recursively merges defaults, rejects unknown options, validates numeric ranges and task/dataset combinations, and checks the model registry before it reads the dataset. An unknown model reports the registered model names. For example, `batch_size: -1` and `model.name: abc` fail before a training run starts.

Each invocation creates a new directory such as `outputs/experiment_001/`:

```text
outputs/experiment_001/
  config.yaml
  best_model.pth
  last_model.pth
  metrics.json
  predictions.csv
  plots/
  logs/
    train.log
```

`config.yaml` records the merged configuration, including CLI overrides and the class count inferred from the dataset. `metrics.json` records model and dataset details, epoch history, evaluation metrics, and checkpoint filenames. Existing experiment directories are not overwritten. `torch.manual_seed`, Python's random seed, and deterministic cuDNN settings are set from `training.seed`; bit-for-bit reproduction can still depend on PyTorch, CUDA, hardware, and data versions.

## Key terms

- **Config (configuration)**: the complete set of values that controls a training run. Keeping it in YAML lets a user change a model, dataset, or output path without editing Python.
- **Hyperparameter**: a value chosen before or during training, such as learning rate, batch size, number of epochs, or weight decay. It affects how the model learns but is not itself learned from the training examples.
- **Experiment**: one particular training run with a specific config, model, dataset, metrics, and checkpoint. UATF saves each run in its own local directory.
- **Reproducibility**: keeping enough of the run inputs and results to understand or repeat an experiment. UATF stores the effective config, dataset summary, seed, metric history, and model checkpoint together. Exact numerical identity is not guaranteed across different environments.

AI experiments need configuration management because a result only has meaning alongside the settings and data that produced it. Saved configs make comparisons understandable, reduce accidental changes between runs, and allow the same workflow to run from the CLI or later from another interface. This phase uses ordinary local files; it does not add a database or an experiment-tracking service.
