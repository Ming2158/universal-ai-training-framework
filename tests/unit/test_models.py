from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn
from torchvision import models

from uatf.config import load_config
from uatf.models import (
    LOSS_REGISTRY,
    MODEL_ADAPTERS,
    MODEL_REGISTRY,
    build_model_from_config,
    get_loss,
    get_model,
    get_model_adapter,
    register_loss,
    register_model,
)


def test_resnet50_single_label_forward_and_loss() -> None:
    model = get_model("resnet50", num_classes=3).eval()
    images = torch.randn(2, 3, 32, 32)
    targets = torch.tensor([0, 2], dtype=torch.long)

    with torch.no_grad():
        logits = model(images)
    loss = get_loss("classification")(logits, targets)

    assert logits.shape == (2, 3)
    assert torch.isfinite(loss)
    assert isinstance(get_loss("classification"), nn.CrossEntropyLoss)


def test_resnet50_multi_label_forward_and_loss() -> None:
    label_count = 14
    model = get_model("resnet50", num_classes=label_count, task="multi_label").eval()
    images = torch.randn(2, 3, 32, 32)
    targets = torch.randint(0, 2, (2, label_count), dtype=torch.float32)

    with torch.no_grad():
        logits = model(images)
    loss = get_loss("multi_label_classification")(logits, targets)

    assert logits.shape == (2, label_count)
    assert torch.isfinite(loss)
    assert isinstance(get_loss("multi_label"), nn.BCEWithLogitsLoss)


def test_pretrained_resnet50_uses_default_weights_and_replaces_head(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_resnet50(*, weights: object) -> nn.Module:
        captured["weights"] = weights
        model = nn.Module()
        model.fc = nn.Linear(8, 1000)
        return model

    monkeypatch.setattr(models, "resnet50", fake_resnet50)
    model = get_model("resnet50", num_classes=5, pretrained=True)

    assert captured["weights"] is models.ResNet50_Weights.DEFAULT
    assert model.fc.out_features == 5


def test_build_model_from_yaml_config(tmp_path: Path) -> None:
    config_path = tmp_path / "model.yaml"
    config_path.write_text(
        "task:\n  type: multi_label_classification\n"
        "model:\n  name: resnet50\n  pretrained: false\n  num_classes: 14\n",
        encoding="utf-8",
    )

    model = build_model_from_config(load_config(config_path)).eval()
    with torch.no_grad():
        output = model(torch.randn(1, 3, 32, 32))

    assert output.shape == (1, 14)


def test_convnext_replaces_classifier_head() -> None:
    model = get_model("convnext", num_classes=4)
    assert model.classifier[-1].out_features == 4


def test_efficientnet_replaces_classifier_head() -> None:
    model = get_model("efficientnet", num_classes=4)
    assert model.classifier[-1].out_features == 4


def test_vit_replaces_classifier_head() -> None:
    model = get_model("vit", num_classes=4)
    assert model.heads.head.out_features == 4


def test_model_registry_accepts_new_builder() -> None:
    name = "test_identity"
    register_model(name, lambda output_count, pretrained: nn.Linear(2, output_count))
    try:
        model = get_model(name, num_classes=7)
        assert model.out_features == 7
    finally:
        del MODEL_REGISTRY[name]


def test_model_adapters_describe_builtin_capabilities() -> None:
    adapter = get_model_adapter("vit")

    assert adapter.identifier == "vit"
    assert adapter.display_name == "ViT-B/16"
    assert adapter.supported_tasks == ("classification", "multi_label")
    assert adapter.supports_pretrained is True
    assert adapter.pretrained_source == "torchvision ViT_B_16_Weights.DEFAULT"
    assert adapter.input_size == (224, 224)
    assert adapter.input_size_required is True
    assert adapter.classifier_head == "heads.head"
    assert adapter.pretrained_cache_path is not None


def test_pretrained_missing_cache_fails_before_download_in_offline_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(torch.hub, "get_dir", lambda: str(tmp_path / "torch-cache"))
    monkeypatch.setenv("UATF_OFFLINE", "1")

    with pytest.raises(RuntimeError, match="not cached.*No random-weight fallback"):
        get_model("resnet50", num_classes=2, pretrained=True)


def test_pretrained_load_errors_are_not_silently_replaced_by_random_weights(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_load(*, weights: object) -> nn.Module:
        raise OSError("network unavailable")

    monkeypatch.delenv("UATF_OFFLINE", raising=False)
    monkeypatch.setattr(models, "resnet50", fail_load)

    with pytest.raises(RuntimeError, match="torchvision ResNet50_Weights.DEFAULT.*No random-weight fallback"):
        get_model("resnet50", num_classes=2, pretrained=True)


def test_registered_adapter_rejects_unsupported_task_and_pretraining() -> None:
    name = "classification_only_test"
    register_model(
        name,
        lambda output_count, pretrained: nn.Linear(2, output_count),
        supported_tasks=("classification",),
        supports_pretrained=False,
    )
    try:
        with pytest.raises(ValueError, match="does not support task 'multi_label'"):
            get_model(name, num_classes=2, task="multi_label")
        with pytest.raises(ValueError, match="does not support pretrained weights"):
            get_model(name, num_classes=2, pretrained=True)
    finally:
        MODEL_REGISTRY.pop(name, None)
        MODEL_ADAPTERS.pop(name, None)


def test_custom_loss_registry_checks_task_contract() -> None:
    name = "test_custom_loss"
    register_loss(
        name,
        nn.MSELoss,
        supported_tasks=("classification",),
        input_contract="scores",
        target_contract="class-index scores",
    )
    try:
        assert isinstance(get_loss("classification", name), nn.MSELoss)
        with pytest.raises(ValueError, match="does not support task 'multi_label'"):
            get_loss("multi_label", name)
    finally:
        LOSS_REGISTRY.pop(name, None)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_resnet50_forwards_on_cuda() -> None:
    device = torch.device("cuda:0")
    model = get_model("resnet50", num_classes=2).to(device).eval()
    images = torch.randn(1, 3, 32, 32, device=device)

    with torch.no_grad():
        output = model(images)

    assert output.device == device
    assert output.shape == (1, 2)


def test_unknown_model_and_task_fail_with_actionable_errors() -> None:
    with pytest.raises(ValueError, match="Available models"):
        get_model("not-a-model", num_classes=2)
    with pytest.raises(ValueError, match="Unsupported task"):
        get_loss("object_detection")
