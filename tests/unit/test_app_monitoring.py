from __future__ import annotations

import io
import json
from pathlib import Path

import pytest
from PIL import Image

from app import app
from app.app import _chart_frame, _format_duration
from app.monitoring import format_resource_snapshot
from uatf.tasks import TaskManager, get_task


def test_gui_consumes_live_events_and_preserves_epoch_curves(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset_path = tmp_path / "dataset"
    for class_name in ("cats", "dogs"):
        class_dir = dataset_path / class_name
        class_dir.mkdir(parents=True)
        for index in range(3):
            Image.new("RGB", (10, 8), (40 + index, 80, 120)).save(
                class_dir / f"{index}.png"
            )
    monkeypatch.setattr(app, "DATASET_REGISTRY", tmp_path / "registry")
    monkeypatch.setattr(app, "OUTPUT_ROOT", tmp_path / "runs")
    monkeypatch.setattr(app, "CONFIG_ROOT", tmp_path / "configs")
    monkeypatch.setattr(app, "HardwareResourceSampler", lambda: _FakeSampler())
    _, _, dataset_state = app._inspect_selected_dataset(
        None, str(dataset_path), "classification", "image_folder", 0.2, 42, 224
    )

    class FakeProcess:
        pid = 4242
        stdout = io.StringIO("child stdout\n")

        def poll(self) -> int:
            return 0

        def wait(self) -> int:
            return 0

    def fake_popen(command: list[str], **_kwargs: object) -> FakeProcess:
        events_path = Path(command[command.index("--events-file") + 1])
        experiment = tmp_path / "runs" / "experiment_001"
        (experiment / "logs").mkdir(parents=True)
        (experiment / "logs" / "train.log").write_text(
            "2026-10-13 12:00:00 INFO uatf.training: started\n", encoding="utf-8"
        )
        (experiment / "best_model.pth").write_bytes(b"checkpoint")
        (experiment / "metrics.json").write_text(
            json.dumps(
                {
                    "task": "classification",
                    "model": {"name": "test"},
                    "dataset": {},
                    "training": {},
                    "evaluation": {"primary_metric": "accuracy", "primary_value": 1.0},
                    "environment": {},
                }
            ),
            encoding="utf-8",
        )
        events = [
            {
                "event_type": "training_started",
                "experiment_dir": str(experiment),
                "total_epochs": 1,
            },
            {
                "event_type": "training_progress",
                "epoch": 1,
                "total_epochs": 1,
                "stage": "training",
                "batch_index": 1,
                "total_batches": 2,
                "metrics": {"train_loss": 0.8, "learning_rate": 0.01},
                "timestamp_utc": "2026-10-13T04:00:00+00:00",
            },
            {
                "event_type": "training_progress",
                "epoch": 1,
                "total_epochs": 1,
                "stage": "validation",
                "batch_index": 1,
                "total_batches": 1,
                "metrics": {"val_loss": 0.4, "val_accuracy": 1.0},
                "timestamp_utc": "2026-10-13T04:00:01+00:00",
            },
            {
                "event_type": "epoch_completed",
                "epoch": 1,
                "total_epochs": 1,
                "metrics": {
                    "epoch": 1.0,
                    "train_loss": 0.7,
                    "val_loss": 0.4,
                    "val_accuracy": 1.0,
                    "learning_rate": 0.01,
                    "time_seconds": 2.0,
                },
                "timestamp_utc": "2026-10-13T04:00:02+00:00",
            },
            {"event_type": "training_completed", "experiment_dir": str(experiment)},
        ]
        events_path.write_text(
            "".join(json.dumps(event) + "\n" for event in events), encoding="utf-8"
        )
        return FakeProcess()

    monkeypatch.setattr(app.subprocess, "Popen", fake_popen)
    results = list(
        app.start_training(
            None,
            str(dataset_path),
            dataset_state,
            "classification",
            "image_folder",
            0.2,
            42,
            224,
            True,
            "resnet50",
            1,
            2,
            0.001,
            "cpu",
        )
    )

    completed = results[-1]
    assert len(completed) == 25
    assert completed[0] == "训练完成"
    assert completed[1] == 100
    assert completed[7] == "0.7000"
    assert completed[8] == "0.4000"
    assert completed[9] == "accuracy: 1.0000"
    assert completed[10] == "accuracy: 1.0000"
    assert completed[11] == "1"
    assert set(completed[14]["series"]) == {"Train Loss", "Validation Loss"}
    assert completed[15]["value"].tolist() == [1.0]
    assert completed[16]["value"].tolist() == [0.01]
    assert completed[21]["evaluation"]["primary_value"] == 1.0
    assert "started" in completed[17]
    assert completed[18].endswith("train.log")
    assert completed[23].endswith("best_model.pth")
    assert len(completed[24]) == 32


def test_gui_can_query_and_stop_a_persisted_task(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_dir = tmp_path / "runs"
    experiment_dir = output_dir / "experiment_007"
    log_path = experiment_dir / "logs" / "train.log"
    log_path.parent.mkdir(parents=True)
    log_path.write_text("training started\n", encoding="utf-8")
    monkeypatch.setattr(app, "OUTPUT_ROOT", output_dir)

    manager = TaskManager(output_dir, "refresh-me")
    manager.acquire()
    manager.update("Preparing")
    manager.set_experiment_directory(experiment_dir)
    manager.set_stage("training", epoch=2, total_epochs=5)
    manager.set_stage("validation", epoch=2, total_epochs=5)

    restored = app.refresh_task_status()
    assert restored[0].startswith("验证中 | 任務 refresh-me")
    assert restored[1] == pytest.approx(30)
    assert restored[2] == "2 / 5"
    assert "training started" in restored[3]
    assert restored[7] == "refresh-me"

    message = app.request_training_stop("refresh-me")
    assert "正在停止" in message
    assert get_task(output_dir, "refresh-me")["status"] == "Stopping"
    manager.update("Stopped", stage="stopped")
    manager.release()


class _FakeSampler:
    def attach_process(self, _process_id: int) -> None:
        return None

    def snapshot(self) -> dict[str, object]:
        return {
            "gpu": {
                "name": None,
                "utilization_percent": None,
                "memory_used_bytes": None,
                "memory_total_bytes": None,
                "torch_allocated_bytes": None,
                "temperature_c": None,
                "power_watts": None,
            },
            "cpu": {"system_percent": None, "training_process_percent": None},
            "memory": {
                "used_bytes": None,
                "available_bytes": None,
                "total_bytes": None,
                "training_process_rss_bytes": None,
            },
        }


def test_resource_format_marks_missing_hardware_values_as_unavailable() -> None:
    snapshot = {
        "gpu": {
            "name": None,
            "utilization_percent": None,
            "memory_used_bytes": None,
            "memory_total_bytes": None,
            "torch_allocated_bytes": None,
            "temperature_c": None,
            "power_watts": None,
        },
        "cpu": {"system_percent": None, "training_process_percent": None},
        "memory": {
            "used_bytes": None,
            "available_bytes": None,
            "total_bytes": None,
            "training_process_rss_bytes": None,
        },
    }

    formatted = format_resource_snapshot(snapshot)

    assert "GPU: 不可用" in formatted
    assert "利用率 不可用" in formatted
    assert "显存 不可用" in formatted
    assert "系统内存: 不可用" in formatted
    assert "不可用" in formatted


def test_training_dashboard_helpers_preserve_missing_and_elapsed_values() -> None:
    frame = _chart_frame(
        [{"epoch": 1, "series": "Train Loss", "value": 0.75}]
    )

    assert list(frame.columns) == ["epoch", "series", "value"]
    assert frame.iloc[0].to_dict() == {
        "epoch": 1,
        "series": "Train Loss",
        "value": 0.75,
    }
    assert _format_duration(None) == "计算中"
    assert _format_duration(65) == "1:05"
    assert _format_duration(3665) == "1:01:05"
