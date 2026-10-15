from __future__ import annotations

import yaml

from app import app


def test_copied_configuration_launches_under_a_fresh_output_root(
    tmp_path, monkeypatch
) -> None:
    output_root = tmp_path / "ui-runs"
    config_root = tmp_path / "ui-configs"
    captured = {}

    def fake_popen(command, **kwargs):
        captured["command"] = command
        captured["kwargs"] = kwargs
        return object()

    monkeypatch.setattr(app, "OUTPUT_ROOT", output_root)
    monkeypatch.setattr(app, "CONFIG_ROOT", config_root)
    monkeypatch.setattr(app.subprocess, "Popen", fake_popen)
    configuration = {
        "task": {"type": "classification"},
        "dataset": {"path": str(tmp_path / "dataset")},
        "training": {"epochs": 3},
        "output": {"directory": str(tmp_path / "old-runs"), "experiment_name": "old-name"},
    }

    status, task_id = app.launch_copied_configuration(
        yaml.safe_dump(configuration, sort_keys=False)
    )

    assert task_id
    saved_configuration = yaml.safe_load((config_root / f"{task_id}.yaml").read_text())
    assert saved_configuration["output"]["directory"] == str(output_root)
    assert "experiment_name" not in saved_configuration["output"]
    assert captured["command"][-1] == task_id
    assert captured["kwargs"]["cwd"] == app.PROJECT_ROOT
    assert "已启动新训练任务" in status


def test_experiment_browser_filters_and_selects_saved_runs(tmp_path, monkeypatch) -> None:
    directory = tmp_path / "experiment_001"
    directory.mkdir()
    (directory / "experiment.json").write_text(
        '{"id":"run-1","name":"birds","status":"Completed",'
        '"created_at_utc":"2026-10-14T00:00:00+00:00","model":{"name":"resnet50"}}',
        encoding="utf-8",
    )
    monkeypatch.setattr(app, "OUTPUT_ROOT", tmp_path)

    frame, selected, compared = app.refresh_experiment_browser(model_filter="resnet")
    unfiltered, _, _ = app.refresh_experiment_browser(None, None, None, None, None)

    assert frame.loc[0, "id"] == "run-1"
    assert selected["value"] == "run-1"
    assert compared["choices"][0][1] == "run-1"
    assert unfiltered.loc[0, "id"] == "run-1"
