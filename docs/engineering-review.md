# UATF Engineering Review

Review date: 2026-10-08  
Reviewed revision: `039df8b` (`feat: add Gradio training interface`)  
Working tree at review start: clean

## Executive Summary

The repository has a working local image-classification training flow for single-label and multi-label tasks. Dataset adapters, model construction, training, evaluation, YAML configuration, CLI, and Gradio UI are separated into understandable modules. Existing tests and end-to-end smoke runs pass when the retained project packages are loaded by the Codex Python runtime.

The main release gaps are missing configuration and CLI integration tests, print-based production training logs, incomplete architecture and troubleshooting documentation, and stale README structure/roadmap text. No performance tuning is supported by the available measurements, so DataLoader and AMP defaults should remain conservative.

## Repository and Git

- The current branch is `feature/model-system`; `HEAD` and `origin/main` point to `039df8b`. Local `main` points to `c81a22a`, before the model, training, evaluation, configuration, and UI work.
- History contains seven phase-oriented commits. No version tags or release changelog existed at review time.
- Tracked files include only source, configuration, documentation, and placeholder files. Datasets, outputs, virtual environments, bytecode, and package build metadata are ignored.
- Existing ignored user data and outputs were present before this review and were preserved.

## Architecture and Dependencies

The runtime flow is `uatf.data` -> `uatf.models` -> `uatf.training` -> `uatf.evaluation`, coordinated by `uatf.cli`. `app.app` launches the same CLI in a subprocess. Configuration validation lives in `uatf.config`; YAML defaults and project dependencies are declared in `pyproject.toml`.

Module boundaries are generally clear and there is no obvious dependency cycle. `uatf/tasks`, `uatf/utils`, and `uatf/ui` do not contain implementations; the README incorrectly lists tasks and utils as existing modules. Task-specific behavior is currently implemented as classification/multi-label branches in config, model, loss, evaluation, and UI code.

The dependency set is small and appropriate for this scope: PyTorch, torchvision, Pillow, PyYAML, Matplotlib, and Gradio. Pytest is the only declared development dependency. No linter or type checker is configured.

## Code Quality and Error Handling

- Dataset adapters validate CSV structure, labels, duplicate paths, missing files, and unreadable images with `DatasetValidationError`.
- Configuration validation rejects unknown fields and invalid ranges before dataset loading.
- Trainer checks batch shapes, scalar/finite loss and metrics, CUDA availability, and CUDA out-of-memory failures.
- Checkpoint loading validates model keys and shapes, but malformed optimizer/scaler state can still surface lower-level exceptions. Checkpoint writes use a temporary file and replace, but write failures are not wrapped with checkpoint-specific context.
- Trainer and CLI training status use `print` plus stdout/stderr redirection. Logs have no severity levels or standard timestamps, and failures are appended as a separate traceback block.
- There is some repeated CSV image validation and repeated model-head replacement code, but neither currently justifies a broad refactor.
- Types are present throughout the public flow, while several loader/dataset and callback annotations remain broad. No static type checker is configured.

## Tests and Baseline Verification

There are 37 tests across dataset, model, evaluation, and training-engine integration modules. There were no tests for config validation, CLI orchestration, or the UI callback. The training integration tests cover validation, checkpoint save/reload, resume, early stopping, invalid batches, non-finite loss, and CUDA AMP.

The checked-in virtual environments point to a removed Python 3.12.10 installation, so their `python.exe` launchers cannot start. The Codex Python 3.12.14 runtime was able to import the packages retained under `.venv/Lib/site-packages` and was used to collect this baseline:

- Existing suite: 37 passed.
- CUDA smoke test: passed on NVIDIA GeForce RTX 5080 Laptop GPU, CUDA runtime 12.8.
- Gradio UI: constructed successfully; local server returned HTTP 200.
- MNIST: one CUDA epoch completed and wrote best/last checkpoints, metrics, predictions, and plots.
- `configs/example.yaml`: one-epoch multi-label CUDA run completed with a temporary 12-image fixture and wrote an experiment report and checkpoint.
- Ruff is not installed, and the project has no lint configuration; a lint run was unavailable.

The model smoke scores are only evidence that the code path executes; the tiny temporary dataset is not a quality benchmark.

## Configuration and Experiment Traceability

The CLI saves the merged effective config, UTC start/completion timestamps, Python/PyTorch/device metadata, task/model metadata, dataset format/path/count/class names, training history, evaluation metrics, and best/last checkpoint names. This is a useful local experiment record. It does not identify the exact dataset contents with a manifest or checksum, so moving or modifying data at the same path is not fully detectable from the report.

Defaults are represented both in `uatf.config.DEFAULT_CONFIG` and `configs/default.yaml`, which can drift. The example config references `datasets/example`, which is intentionally not tracked and must be supplied by the user.

## Performance

`num_workers` is configurable and defaults to zero. DataLoaders do not enable `pin_memory`, and tensor transfers are synchronous. These are conservative settings and are not defects without a representative workload measurement. The available GPU checks establish CUDA operation, not utilization or throughput. Keep the current defaults until profiling a representative dataset/model pair; AMP is opt-in and has a CUDA integration test.

## Documentation and Release Readiness

The README covers installation, usage, dataset formats, model choices, training, evaluation, and the Gradio UI. Phase 3-9 documents provide useful detail. The README's module tree is stale, its status and roadmap contain outdated TODOs, and there is no concise architecture overview or troubleshooting reference. No `CHANGELOG` or versioned release notes existed. `pyproject.toml` still reports version `0.1.0`.

## Recommended Phase 10 Work

1. Add config unit tests and a small CLI integration test that exercises config, dataset, model, training, evaluation, and experiment outputs together.
2. Replace production Trainer/CLI status prints and output redirection with standard logging while keeping the CLI's readable epoch lines for the UI.
3. Improve checkpoint write/read errors with path-specific context and add focused tests for malformed or unwritable checkpoints.
4. Correct the README, document the actual module layout, and add architecture and troubleshooting guides.
5. Add a v1.0.0 changelog and release notes, set package metadata to `1.0.0`, and run release hygiene checks before creating a local tag.
6. Keep `num_workers`, `pin_memory`, and AMP defaults unchanged until representative profiling provides evidence for a change.

The local `main` branch is behind the reviewed feature branch. A release should be based on the completed and reviewed changes, then fast-forward `main` before tagging. Do not push automatically.
