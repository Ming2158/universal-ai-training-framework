# Changelog

Notable changes for each release are recorded here.

## [Unreleased]

### Added

- A bilingual interactive CLI wizard for choosing a task, dataset, model, and training settings.
- A Windows launcher and a usage guide for GUI and CLI workflows.

### Fixed

- The interactive CLI wizard now works with legacy GBK Windows consoles.

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
