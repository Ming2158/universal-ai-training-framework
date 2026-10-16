# UATF V2.1.0 Release Notes

**Release date:** 2026-10-16

## Summary

V2.1.0 fixes the GUI training-start validation path and makes fixed-size image resizing optional for convolutional models. Resizing stays lazy in the DataLoader; UATF does not create a second resized copy of the dataset.

## Training and data pipeline

- Invalid UI settings and stale dataset inspection state are rejected before a training task is submitted. A rejected request stays on the current page and displays its error.
- The dataset page has a fixed-size resize toggle. When disabled, images retain their original dimensions and each batch is padded to its largest image, rounded up to a multiple of 32. This mode is available for ResNet50, ConvNeXt-Tiny, and EfficientNet-B0. ViT-B/16 requires fixed-size resize.
- The selected preprocessing setting is saved with the experiment and model asset and is reused for evaluation and inference. Checkpoint resume validates the preprocessing setting while accepting older checkpoints that predate it.
- DataLoader workers are configurable from 0 to 16. The GUI defaults to at most four workers based on available CPU cores. Worker processes stay alive between epochs; CUDA runs use pinned memory and non-blocking device copies.
- Image-folder inspection reports nested split folders during the existing traversal instead of scanning the tree a second time.

## ZIP imports

- The uncompressed archive limit is 128 GiB, still bounded by the 50,000-member limit.
- Before extraction, UATF checks that the destination volume has room for the expanded archive plus a 2 GiB reserve.

## Compatibility and limits

- Configurations without `dataset.resize_enabled` continue to resize to their configured `dataset.image_size` (224 by default).
- Disabling resize can increase CPU/GPU memory use substantially, especially with large images and larger batches. Reduce batch size or leave resize enabled if memory is constrained.
- This release was validated with small end-to-end datasets. A representative 50 GB performance benchmark was not available, so throughput improvement depends on image dimensions, storage, CPU, GPU, and worker count.
- ZIP size is also bounded by free space. The UI cannot recover upload data that the browser or operating system fails to write to its temporary upload location.

## Verification

- Full test suite: `132 passed`.
- `ruff check app uatf tests`, `compileall app uatf tests`, and `git diff --check` passed.
- End-to-end CPU flow passed with resizing disabled: variable-size folder training, model-asset evaluation, and variable-size directory inference.
- The actual workspace training callback started its CLI subprocess and persisted a `Completed` task with a best checkpoint after the inspection warning was acknowledged.
- Gradio configuration built successfully with 299 components; the local server returned HTTP 200 and was shut down after the check.
- Browser window activation failed twice with `failed to activate captured window`, so manual browser interaction was not verified.
- A representative 50 GB dataset was not available; no 50 GB throughput claim is made.
