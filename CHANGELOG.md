# Changelog

Notable changes for each release are recorded here.

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
