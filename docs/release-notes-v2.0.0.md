# UATF V2.0.0 Release Notes

**Release date:** 2026-10-16

## Summary

V2.0.0 develops UATF into a more traceable local image-classification workflow while retaining the shared CLI, YAML, Python, and Gradio training path.

## Added

- A configuration-driven Python API for training runs, structured events, stop requests, evaluation, and inference.
- Dataset inspection with image validation, class/label counts, deterministic split reporting, and GUI previews.
- Persistent task records, cooperative stop at a completed epoch, and resume from compatible full training checkpoints.
- Experiment history, model-asset records, artifact references, legacy run browsing, and compatibility-aware experiment comparison.
- Independent evaluation, single-image prediction, batch directory prediction, CSV output, and reloadable PyTorch model exports with JSON sidecars.
- Model adapter metadata and expanded training configuration, including optimizer, scheduling, loss, and augmentation options.
- A Gradio workspace for task monitoring, experiment/model browsing, evaluation, and inference.

## Compatibility

- Existing V1 YAML configurations continue to be accepted by the configuration resolver.
- Existing experiment directories can be read as legacy records without being moved or rewritten.
- Existing CLI config execution and Python component entry points remain available. The `uatf.api` facade is additive.
- Resume requires a compatible full training checkpoint such as `last_model.pth`; exported inference weights are not resume checkpoints.

## Verification

- Full automated suite: **127 passed**, no failures or skips, on Python 3.12.14, PyTorch 2.7.1+cu128, torchvision 0.22.1+cu128, and Gradio 6.29.1.
- Ruff, `compileall`, `pip check`, and `train.py --help` passed.
- Gradio workspace construction passed with 293 components and 41 event dependencies.
- CUDA smoke passed on an NVIDIA GeForce RTX 5080 Laptop GPU with CUDA runtime 12.8. This is a tensor/device smoke check, not a complete GUI training acceptance.
- A 2.0.0 wheel was built with `pip wheel --no-deps --no-build-isolation`; SHA256: `aebaeaaa1822b2581fd1761581796fe642c6b121ec99317e0a3a313abbd7e30f`.
- Browser-level acceptance of GUI stop/resume, refreshed history/assets, evaluation, inference, downloads, restart persistence, and responsive widths remains incomplete. See the [release acceptance record](development/v2.0.0-release-acceptance.md).

## Known limits

- UATF supports single-label and multi-label image classification on one machine. Detection, segmentation, distributed training, and managed experiment storage are outside this release scope.
- Dataset splits are seeded but are not stratified or group-aware; dataset contents are not fully hashed.
- Real pretrained-weight downloads and Python 3.10/3.11 environments were not verified in this release run.
- GUI workflow coverage is incomplete at the browser interaction level; the acceptance record separates callback and backend test evidence from manual browser evidence.
