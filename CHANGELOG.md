# Changelog

Notable changes for each release are recorded here.

## [Unreleased]

## [2.0.0] - 2026-10-16

### Added

- A bilingual interactive CLI wizard for choosing a task, dataset, model, and training settings.
- A Windows launcher and a usage guide for GUI and CLI workflows.
- A unified Python API and persistent task lifecycle with cooperative stop and checkpoint resume.
- Dataset inspection, experiment and model-asset history, independent evaluation, and image inference.
- Gradio progress monitoring, experiment browsing, model asset management, evaluation, and inference views.
- Expanded training configuration with model adapters, optimizer and scheduler options, augmentation, and task-aware losses.

### Compatibility

- Existing UATF 1.0.0 YAML configurations and historical experiment directories remain readable.
- New experiment records and model assets are stored alongside the existing local run directory layout.
- The existing CLI configuration mode, Python components, and V1.0.0 release tag remain available and unchanged.

### Fixed

- The interactive CLI wizard now works with legacy GBK Windows consoles.

### Verification

- See [V2.0.0 release notes](docs/release-notes-v2.0.0.md) and the [release acceptance record](docs/development/v2.0.0-release-acceptance.md) for verification scope and known gaps.

## [1.0.0] - 2026-10-08

### Added

- Standard INFO, WARNING, ERROR, and DEBUG logging for config-driven training runs.
- Configuration unit tests and an end-to-end CLI integration test.
- Checkpoint diagnostics for write failures and malformed optimizer, scheduler, scaler, and early-stopping state.
- Architecture and troubleshooting documentation, plus versioned release notes.

### Fixed

- CLI folder classification now accepts `dataset.path` pointing directly to class directories.

### Included

- Single-label and multi-label image classification datasets and evaluation.
- ResNet50, ConvNeXt-Tiny, EfficientNet-B0, and ViT-B/16 model choices.
- Config-driven CLI, reusable Trainer, checkpoints and resume API, optional AMP, and local Gradio UI.
