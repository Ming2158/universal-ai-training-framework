# Phase 5: Model System

UATF builds image models in `uatf.models`. The training loop can accept any returned PyTorch model and does not need to know which architecture produced it. Dataset batches already follow the matching target conventions: single-label targets are integer class IDs with batch shape `[B]`; multi-label targets are float multi-hot vectors with batch shape `[B, L]`.

## Create a model

```python
from uatf.models import get_loss, get_model

model = get_model("resnet50", num_classes=3, pretrained=False)
loss_function = get_loss("classification")
```

For multi-label classification, the output count is the number of independent labels:

```python
model = get_model(
    "resnet50",
    num_classes=14,
    pretrained=True,
    task="multi_label",
)
loss_function = get_loss("multi_label")
```

The model returns raw logits with shape `[batch_size, num_classes]`. The model does not apply softmax or sigmoid. `CrossEntropyLoss` expects integer class IDs and internally handles the single-label class probabilities. `BCEWithLogitsLoss` compares each independent label logit with a float target of shape `[batch_size, num_labels]`; it combines sigmoid and binary cross-entropy in a numerically stable operation.

## Create from YAML

```yaml
task:
  type: multi_label_classification

model:
  name: resnet50
  pretrained: true
  num_classes: 14
```

```python
from uatf.config import load_config
from uatf.models import build_model_from_config, get_loss

config = load_config("configs/default.yaml")
model = build_model_from_config(config)
loss_function = get_loss(config["task"]["type"])
```

The supported model choices are `resnet50`, `convnext`, `efficientnet`, and `vit`. They use ResNet50, ConvNeXt-Tiny, EfficientNet-B0, and ViT-B/16 respectively. Their task support, pretrained source, input size, classifier head, and checkpoint/inference contract are available as `ModelAdapter` metadata. See the [Phase 7 model and training guide](phase7-model-training.md) for the metadata and offline weight behavior.

## Model terms

- **Model:** the full neural network that maps an input image tensor to output logits.
- **Backbone:** the feature extraction layers that learn visual patterns such as edges, textures, and shapes. In ResNet50 this is everything before `fc`; in ConvNeXt-Tiny it is the `features` stack plus classifier normalization and flattening.
- **Classifier head:** the final layer that maps learned image features to one score per class or label. UATF replaces the original ImageNet head with a new linear layer sized for the current task.
- **Pretrained:** initialized from weights learned earlier, here the torchvision ImageNet DEFAULT weights. Enabling this may download the weights on first use; set `UATF_OFFLINE=1` to reject an uncached checkpoint before a download attempt.
- **Transfer learning:** reusing a pretrained backbone for a new task and replacing its output head. It can reduce the amount of task data and training time needed compared with learning all visual features from random initialization.
- **Fine-tuning:** continuing training on task-specific data. The new head starts with fresh parameters; by default, backbone parameters remain trainable too, so the whole network can adapt.
- **Logits:** raw, unnormalized output scores. They are the correct input to the selected PyTorch loss; apply sigmoid later when multi-label probabilities are needed for inference, or softmax for single-label probabilities.
- **Output shape:** `[B, C]` for `B` images and `C` output classes/labels. In multi-label classification each of the `C` values is an independent decision, not one choice among `C` mutually exclusive classes.

## Checks

The model unit tests cover model construction, forward output shapes, both loss types, YAML-based creation, pretrained weight selection without downloading weights, and an optional CUDA forward pass. Run them from the project root with:

```powershell
python -m pytest tests/unit/test_models.py
```

The CUDA test is skipped when CUDA is unavailable. Tests construct models without pretrained weights, so they do not require a network connection.
