from __future__ import annotations

import io
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from uatf import cli_wizard


def test_multilabel_wizard_runs_with_gbk_console(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = tmp_path / "dataset"
    (dataset / "images").mkdir(parents=True)
    (dataset / "annotations.csv").write_text("image_path,cat\n", encoding="utf-8")
    monkeypatch.setattr(cli_wizard, "PROJECT_ROOT", tmp_path)
    output = io.BytesIO()
    console = io.TextIOWrapper(output, encoding="gbk")
    monkeypatch.setattr(sys, "stdout", console)

    answers = ["2", str(dataset), "", "", "", "", "", "", ""]
    with patch("builtins.input", side_effect=answers):
        result = cli_wizard.run_wizard(language="zh")

    assert result is not None
    configuration, config_path, language = result
    console.flush()

    assert language == "zh"
    assert configuration["task"]["type"] == "multi_label"
    assert configuration["dataset"]["format"] == "csv_multilabel"
    assert "optimizer" not in configuration["training"]
    saved_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert saved_config["task"]["type"] == "multi_label"
    assert "optimizer" not in saved_config["training"]
    assert "训练设置确认" in output.getvalue().decode("gbk")
