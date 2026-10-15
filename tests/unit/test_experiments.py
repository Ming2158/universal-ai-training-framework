from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
import yaml
from torch import nn

from uatf.experiments import (
    compare_experiments,
    export_model_asset,
    get_experiment,
    list_experiments,
    list_model_assets,
    load_model_asset,
    register_model_asset,
)
from uatf.models.factory import MODEL_REGISTRY


def test_legacy_runs_remain_visible_and_filterable(tmp_path: Path) -> None:
    directory = tmp_path / "experiment_legacy"
    directory.mkdir()
    (directory / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "task": {"type": "classification"},
                "model": {"name": "resnet50"},
                "evaluation": {"primary_metric": "accuracy"},
            }
        ),
        encoding="utf-8",
    )
    metrics = {
        "experiment": {"name": "cats-v1", "started_at_utc": "2026-10-01T00:00:00+00:00"},
        "task": "classification",
        "model": {"name": "resnet50"},
        "evaluation": {"primary_metric": "accuracy", "primary_value": 0.9},
    }
    (directory / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")

    rows = list_experiments(tmp_path, name="cats", model="resnet", status="Completed")

    assert len(rows) == 1
    assert rows[0]["legacy"] is True
    assert rows[0]["best_metric"] == 0.9
    assert get_experiment(tmp_path, "experiment_legacy")["metrics"] == metrics


def test_legacy_best_checkpoint_is_available_as_a_read_only_asset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def build_tiny(output_count: int, pretrained: bool) -> nn.Module:
        assert pretrained is False
        return nn.Sequential(nn.Flatten(), nn.Linear(12, output_count))

    monkeypatch.setitem(MODEL_REGISTRY, "legacy_tiny", build_tiny)
    directory = tmp_path / "experiment_old"
    directory.mkdir()
    config = {
        "task": {"type": "classification"},
        "model": {"name": "legacy_tiny", "num_classes": 2, "pretrained": False},
    }
    (directory / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    model = build_tiny(2, False)
    torch.save({"model_state_dict": model.state_dict()}, directory / "best_model.pth")

    assets = list_model_assets(tmp_path)
    loaded = load_model_asset(tmp_path, "legacy-model-experiment_old")

    assert assets[0]["legacy"] is True
    assert assets[0]["weights_kind"] == "legacy_full_training_checkpoint"
    assert isinstance(loaded, nn.Module)


def test_comparison_warns_when_validation_data_differs(tmp_path: Path) -> None:
    for index, split_id in enumerate(("split-a", "split-b"), start=1):
        directory = tmp_path / f"experiment_{index:03d}"
        directory.mkdir()
        (directory / "config.yaml").write_text(
            yaml.safe_dump(
                {
                    "task": {"type": "classification"},
                    "model": {"name": "resnet50", "pretrained": False},
                    "training": {"epochs": 1, "learning_rate": 0.01},
                    "evaluation": {"primary_metric": "accuracy", "threshold": 0.5},
                }
            ),
            encoding="utf-8",
        )
        (directory / "experiment.json").write_text(
            json.dumps(
                {
                    "id": f"run-{index}",
                    "name": directory.name,
                    "status": "Completed",
                    "task": "classification",
                    "dataset": {"id": "same-data", "split_id": split_id},
                    "model": {"name": "resnet50", "pretrained": False},
                    "evaluation": {
                        "primary_metric": "accuracy",
                        "threshold": 0.5,
                        "best_value": 0.9,
                    },
                    "duration_seconds": 1.0,
                }
            ),
            encoding="utf-8",
        )

    result = compare_experiments(tmp_path, ["run-1", "run-2"])

    assert result["comparable"] is False
    assert any("划分不同" in message for message in result["warnings"])
    assert len(result["experiments"]) == 2


def test_model_asset_can_be_loaded_and_exported(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def build_tiny(output_count: int, pretrained: bool) -> nn.Module:
        assert pretrained is False
        return nn.Sequential(nn.Flatten(), nn.Linear(3 * 2 * 2, output_count))

    monkeypatch.setitem(MODEL_REGISTRY, "asset_tiny", build_tiny)
    directory = tmp_path / "experiment_001"
    directory.mkdir()
    configuration = {
        "task": {"type": "classification"},
        "model": {"name": "asset_tiny", "num_classes": 2, "pretrained": False},
    }
    config_path = directory / "config.yaml"
    config_path.write_text(yaml.safe_dump(configuration), encoding="utf-8")
    weights_path = directory / "best_model.pth"
    source_model = build_tiny(2, False)
    torch.save({"model_state_dict": source_model.state_dict(), "optimizer_state_dict": {}}, weights_path)

    asset = register_model_asset(
        tmp_path,
        {
            "id": "model-run-1",
            "name": "run-1",
            "architecture": "asset_tiny",
            "experiment_id": "run-1",
            "weights_path": weights_path,
            "configuration_path": config_path,
            "task": "classification",
        },
    )

    assert asset["weights_path"] == "experiment_001/best_model.pth"
    assert list_model_assets(tmp_path)[0]["available"] is True
    loaded = load_model_asset(tmp_path, "model-run-1")
    assert isinstance(loaded, nn.Module)
    exported = export_model_asset(tmp_path, "model-run-1")
    exported_state = torch.load(exported, map_location="cpu", weights_only=True)
    assert set(exported_state) == set(source_model.state_dict())
    assert exported.with_suffix(exported.suffix + ".json").is_file()

    with pytest.raises(ValueError, match="outside the UATF output directory"):
        register_model_asset(
            tmp_path,
            {
                "id": "outside",
                "weights_path": tmp_path.parent / "outside.pth",
                "configuration_path": config_path,
            },
        )
