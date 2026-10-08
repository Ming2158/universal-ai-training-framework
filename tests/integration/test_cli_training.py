from __future__ import annotations

import json
import runpy
import sys
from pathlib import Path

import pytest
import yaml
from PIL import Image
from torch import nn

from uatf import __version__
from uatf.models.factory import MODEL_REGISTRY


def _build_tiny_model(output_count: int, pretrained: bool) -> nn.Module:
    assert pretrained is False
    return nn.Sequential(nn.Flatten(), nn.Linear(3 * 16 * 16, output_count))


def test_cli_runs_a_classification_experiment_and_records_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setitem(MODEL_REGISTRY, "test_tiny", _build_tiny_model)
    dataset_dir = tmp_path / "images"
    for class_name, color in (("blue", (0, 0, 220)), ("red", (220, 0, 0))):
        class_dir = dataset_dir / class_name
        class_dir.mkdir(parents=True)
        for index in range(4):
            Image.new("RGB", (16, 16), color).save(class_dir / f"{index}.png")

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "dataset": {
                    "path": str(dataset_dir),
                    "validation_fraction": 0.25,
                    "image_size": 16,
                },
                "model": {"name": "test_tiny", "pretrained": False},
                "training": {
                    "epochs": 1,
                    "batch_size": 2,
                    "learning_rate": 0.01,
                    "device": "cpu",
                    "num_workers": 0,
                },
                "output": {"directory": str(tmp_path / "runs")},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(sys, "argv", ["uatf.cli", "--config", str(config_path)])
    with pytest.raises(SystemExit) as exit_info:
        runpy.run_module("uatf.cli", run_name="__main__", alter_sys=True)
    assert exit_info.value.code == 0
    capsys.readouterr()

    experiment_dir = tmp_path / "runs" / "experiment_001"
    report = json.loads((experiment_dir / "metrics.json").read_text(encoding="utf-8"))
    effective_config = yaml.safe_load((experiment_dir / "config.yaml").read_text(encoding="utf-8"))
    log_text = (experiment_dir / "logs" / "train.log").read_text(encoding="utf-8")

    assert report["experiment"]["started_at_utc"]
    assert report["experiment"]["completed_at_utc"]
    assert report["environment"]["uatf"] == __version__
    assert report["dataset"]["train_samples"] + report["dataset"]["validation_samples"] == 8
    assert report["model"]["name"] == "test_tiny"
    assert report["checkpoint"] == {"best": "best_model.pth", "last": "last_model.pth"}
    assert effective_config["model"]["num_classes"] == 2
    assert (experiment_dir / "best_model.pth").is_file()
    assert (experiment_dir / "last_model.pth").is_file()
    assert "INFO uatf.training.trainer: Epoch 1/1" in log_text
    assert "DEBUG uatf.cli: Effective configuration:" in log_text
