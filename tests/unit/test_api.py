from __future__ import annotations

from pathlib import Path

import pytest

import uatf.api
from app.app import build_training_config
from uatf.api import run_training
from uatf.config import load_config, resolve_config
from uatf.tasks import get_latest_task
from uatf.training import TrainingEvent


def test_gui_mapping_yaml_and_python_api_share_the_same_resolution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gui_configuration = build_training_config(
        tmp_path / "dataset",
        "multi_label",
        "resnet50",
        3,
        8,
        0.001,
        "cpu",
    )
    gui_configuration["output"]["directory"] = str(tmp_path / "runs")
    config_path = tmp_path / "gui.yaml"
    config_path.write_text(
        "task:\n"
        "  type: multi_label\n"
        "dataset:\n"
        f"  path: {tmp_path / 'dataset'}\n"
        "  format: csv_multilabel\n"
        "model:\n"
        "  name: resnet50\n"
        "training:\n"
        "  epochs: 3\n"
        "  batch_size: 8\n"
        "  learning_rate: 0.001\n"
        "  device: cpu\n"
        "evaluation:\n"
        "  primary_metric: micro_f1\n"
        "output:\n"
        f"  directory: {tmp_path / 'runs'}\n",
        encoding="utf-8",
    )
    overrides = {"training": {"epochs": 2}}
    captured: dict[str, object] = {}
    expected_path = tmp_path / "run"

    def fake_run(
        effective: dict[str, object],
        source: Path | None,
        *,
        source_configuration: dict[str, object],
        configuration_overrides: dict[str, object],
        event_callback: object,
        stop_requested: object,
        task_status_callback: object,
        task_directory_callback: object,
    ) -> Path:
        captured.update(
            effective=effective,
            source=source,
            source_configuration=source_configuration,
            configuration_overrides=configuration_overrides,
            event_callback=event_callback,
            stop_requested=stop_requested,
            task_status_callback=task_status_callback,
            task_directory_callback=task_directory_callback,
        )
        return expected_path

    monkeypatch.setattr(uatf.api, "run_experiment", fake_run)

    def callback(event: TrainingEvent) -> None:
        return None

    result = run_training(config_path, overrides=overrides, event_callback=callback)

    assert result == expected_path
    assert captured["effective"] == resolve_config(load_config(config_path), overrides)
    assert captured["effective"] == resolve_config(gui_configuration, overrides)
    assert captured["source"] == config_path.resolve()
    assert captured["event_callback"] is callback
    assert callable(captured["stop_requested"])
    assert callable(captured["task_status_callback"])
    assert callable(captured["task_directory_callback"])
    assert get_latest_task(tmp_path / "runs")["status"] == "Completed"


def test_api_rejects_non_mapping_configuration() -> None:
    with pytest.raises(ValueError, match="configuration must be a mapping"):
        run_training(42)  # type: ignore[arg-type]


def test_training_event_has_a_stable_structured_shape() -> None:
    event = TrainingEvent(
        event_type="epoch_completed",
        experiment_dir="runs/experiment_001",
        epoch=1,
        total_epochs=2,
        metrics={"train_loss": 0.4, "val_loss": 0.5},
    )

    assert event.event_type == "epoch_completed"
    assert event.epoch == 1
    assert event.total_epochs == 2
    assert event.metrics == {"train_loss": 0.4, "val_loss": 0.5}
