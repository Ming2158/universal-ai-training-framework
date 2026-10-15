# Phase 7: Model and Training Extensions

Phase 7 extends the existing image-classification pipeline through the shared YAML configuration, `run_training()` Python API, CLI, and Gradio UI. All supported models return raw logits and use the same Trainer, checkpoint, evaluation, and inference implementations.

## Model Adapters

`uatf.models` keeps `MODEL_REGISTRY` as the compatible mapping from model IDs to builders with the `(output_count, pretrained)` signature. `MODEL_ADAPTERS` describes each registered model's display name, task support, pretrained support and source, cache URL, input-size requirement, RGB normalization, classifier head, checkpoint format, inference settings, and limitations. Use `register_model()` for new models and declare their capabilities there.

The built-in IDs remain `resnet50`, `convnext`, `efficientnet`, and `vit`, backed by ResNet-50, ConvNeXt-Tiny, EfficientNet-B0, and ViT-B/16. Each supports single-label and multi-label classification and replaces its final linear classifier with the requested output count. The built-ins use ImageNet RGB normalization by default. The first three accept configurable resized image dimensions; ViT-B/16 requires 224 x 224 input.

Pretrained weights are opt-in through `model.pretrained`. The built-ins use the corresponding torchvision `*_Weights.DEFAULT` ImageNet checkpoint and PyTorch's `hub/checkpoints` cache. A missing checkpoint triggers torchvision's normal download attempt. Set `UATF_OFFLINE=1` to fail before a download when the cache is missing. Download, cache, or state-dict errors are reported with the model, source, and expected cache path; UATF does not retry indefinitely or fall back to random weights. Checkpoints are loaded strictly by torchvision and by the existing UATF model-state loader.

The extension contract preserves the original builder API:

```python
from torch import nn
from uatf.models import register_model

def build_tiny(output_count: int, pretrained: bool) -> nn.Module:
    # Construct the backbone, adapt its classifier head, and return raw logits.
    ...

register_model(
    "tiny",
    build_tiny,
    display_name="Tiny Classifier",
    supported_tasks=("classification",),
    supports_pretrained=False,
    input_size=(224, 224),
    classifier_head="classifier",
    limitations=("Single-label classification only.",),
)
```

New adapters should test a forward pass, training, checkpoint save and reload, evaluation, and inference before they are exposed as supported models. Direct writes to the legacy `MODEL_REGISTRY` remain readable but produce generic capability metadata; prefer `register_model()` for a declared adapter.

## Preprocessing and Augmentation

Resize and RGB normalization remain shared by training, validation, evaluation, and inference. Training augmentation is a separate, optional pipeline. It supports horizontal flip, rotation, brightness, contrast, saturation, and hue changes; every value is validated before data loading. The defaults disable augmentation. Validation, test, independent evaluation, and prediction always use deterministic resize and normalization, even when a model asset's training config records augmentation.

```yaml
dataset:
  augmentation:
    enabled: true
    horizontal_flip_probability: 0.5
    rotation_degrees: 8
    brightness: 0.1
    contrast: 0.1
    saturation: 0.1
    hue: 0.0
```

The Gradio workflow exposes a default-off augmentation switch. CLI, YAML, and Python callers can set each intensity.

## Losses, Metrics, and Training Strategies

The loss registry maps `cross_entropy` to single-label integer targets and raw class logits, and `bce_with_logits` to multi-label float targets and raw label logits. `training.loss.name: auto` selects the task default. Incompatible combinations fail config validation before dataset loading. New registered losses must declare supported tasks and input/target contracts.

`evaluation.primary_metric` is checked against the selected task's metric registry. Final reports include classification accuracy and macro/micro precision, recall, and F1; multi-label reports also include macro and micro ROC-AUC when they are defined. Training monitors remain `val_loss`, `val_accuracy`, or `val_micro_f1` according to task, matching the values the training loop actually computes.

Additional final-evaluation metrics can be registered with `register_metric(name, evaluator, supported_tasks=...)`. The evaluator receives CPU tensors `(targets, predictions)` and must return one finite scalar. Classification predictions are class IDs; multi-label predictions are thresholded binary values. A registered metric can be selected as `evaluation.primary_metric`; training monitors remain limited to values computed during training.

The training config supports Adam, AdamW, and SGD; StepLR and cosine annealing; non-negative weight decay; optional gradient clipping; and gradient accumulation. Defaults preserve the existing Adam + disabled StepLR flow, with no clipping and one optimizer step per batch. The run seeds Python, Torch, CUDA, and the shuffled DataLoader; checkpoints preserve Torch/CUDA and loader generator state. Deterministic results can still vary across hardware and library versions.

```yaml
training:
  loss:
    name: auto
  optimizer:
    name: adamw
    weight_decay: 0.01
  scheduler:
    enabled: true
    name: cosine
    t_max: 10
    eta_min: 0.0
  gradient_accumulation_steps: 2
  gradient_clip_norm: 1.0
```

Gradient clipping and accumulation settings are recorded in checkpoints; resuming with different values is rejected. Legacy checkpoints without these fields remain loadable. The optimizer and scheduler state use PyTorch's standard state dictionaries.

## Dataset Splits

The current random split is seeded, disjoint by sample index, recorded in a split manifest, and reproducible for the same dataset ordering and seed. Existing `train/` plus `validation/` directories are used as supplied and avoid an extra random split.

Stratified splitting is deferred because small or rare classes can make a requested stratified partition impossible, and multi-label stratification needs a separately chosen algorithm and clear failure rules. Grouped splitting is deferred until dataset formats expose a stable group identifier; guessing groups from filenames would risk leakage. Loading a prior split manifest is also deferred; the current manifest records each experiment's exact sample IDs and split for auditing, while explicit train/validation folders let users reuse a split today. These are follow-up capabilities rather than hidden behavior in this phase.

## Scope

Only single-label and multi-label image classification are supported. Object detection, segmentation, distributed execution, multi-GPU scheduling, and hyperparameter search are not implemented. They require task-specific data targets, model outputs, loss/evaluation contracts, and are outside Phase 7.
