from __future__ import annotations

import json
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest
import torch
import yaml
from PIL import Image
from torch import nn

from app import app as gui_app
from uatf import __version__
from uatf.api import run_training
from uatf.cli import main
from uatf.experiments import get_experiment, list_experiments, list_model_assets
from uatf.models.factory import MODEL_REGISTRY
from uatf.tasks import TaskManager, get_latest_task, get_task


def _build_tiny_model(output_count: int, pretrained: bool) -> nn.Module:
    assert pretrained is False
    return nn.Sequential(nn.Flatten(), nn.Linear(3 * 16 * 16, output_count))


class _SubprocessTinyModel(nn.Module):
    def __init__(self, output_count: int, *, delay_seconds: float = 0.0) -> None:
        super().__init__()
        self.network = nn.Sequential(nn.Flatten(), nn.Linear(3 * 16 * 16, output_count))
        self.delay_seconds = delay_seconds

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if self.delay_seconds:
            time.sleep(self.delay_seconds)
        return self.network(images)


def _build_subprocess_tiny_model(output_count: int, pretrained: bool) -> nn.Module:
    assert pretrained is False
    return _SubprocessTinyModel(output_count)


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
    source_configuration = {
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
            "optimizer": {"name": "sgd", "momentum": 0.6},
        },
        "output": {
            "directory": str(tmp_path / "runs"),
            "save_predictions": False,
            "save_plots": False,
        },
    }
    config_path.write_text(
        yaml.safe_dump(source_configuration, sort_keys=False), encoding="utf-8"
    )
    events_path = tmp_path / "events.jsonl"

    exit_code = main(
        [
            "--config",
            str(config_path),
            "--learning-rate",
            "0.02",
            "--events-file",
            str(events_path),
        ]
    )
    assert exit_code == 0
    capsys.readouterr()

    experiment_dir = tmp_path / "runs" / "experiment_001"
    report = json.loads((experiment_dir / "metrics.json").read_text(encoding="utf-8"))
    effective_config = yaml.safe_load((experiment_dir / "config.yaml").read_text(encoding="utf-8"))
    recorded_source_config = yaml.safe_load(
        (experiment_dir / "source_config.yaml").read_text(encoding="utf-8")
    )
    event_rows = [json.loads(line) for line in events_path.read_text(encoding="utf-8").splitlines()]
    log_text = (experiment_dir / "logs" / "train.log").read_text(encoding="utf-8")

    assert report["experiment"]["started_at_utc"]
    assert report["experiment"]["completed_at_utc"]
    assert report["environment"]["uatf"] == __version__
    assert report["dataset"]["train_samples"] + report["dataset"]["validation_samples"] == 8
    assert report["model"]["name"] == "test_tiny"
    assert report["checkpoint"] == {"best": "best_model.pth", "last": "last_model.pth"}
    assert report["experiment"]["configuration_overrides"] == {
        "training": {"learning_rate": 0.02}
    }
    assert effective_config["model"]["num_classes"] == 2
    assert effective_config["training"]["learning_rate"] == 0.02
    assert effective_config["training"]["optimizer"]["name"] == "sgd"
    assert recorded_source_config == source_configuration
    assert event_rows[0]["event_type"] == "training_started"
    assert event_rows[-1]["event_type"] == "training_completed"
    task_record = get_latest_task(tmp_path / "runs")
    assert task_record["status"] == "Completed"
    assert task_record["experiment_id"] == "experiment_001"
    assert task_record["result_directory"] == str(experiment_dir.resolve())
    assert task_record["started_at"] and task_record["finished_at"]
    epoch_events = [event for event in event_rows if event["event_type"] == "epoch_completed"]
    progress_events = [
        event for event in event_rows if event["event_type"] == "training_progress"
    ]
    assert len(epoch_events) == 1
    assert epoch_events[0]["metrics"]["epoch"] == 1.0
    assert {event["stage"] for event in progress_events} == {"training", "validation"}
    assert all(event["timestamp_utc"] for event in event_rows)
    assert (experiment_dir / "best_model.pth").is_file()
    assert (experiment_dir / "last_model.pth").is_file()
    last_checkpoint = torch.load(
        experiment_dir / "last_model.pth", map_location="cpu", weights_only=True
    )
    assert last_checkpoint["optimizer_type"] == "SGD"
    assert last_checkpoint["optimizer_state_dict"]["param_groups"][0]["momentum"] == 0.6
    assert not (experiment_dir / "predictions.csv").exists()
    assert not (experiment_dir / "plots").exists()
    experiment_record = json.loads(
        (experiment_dir / "experiment.json").read_text(encoding="utf-8")
    )
    assert experiment_record["status"] == "Completed"
    assert experiment_record["id"] == report["experiment"]["id"]
    assert experiment_record["dataset"]["id"]
    assert experiment_record["dataset"]["split_id"]
    manifest_path = tmp_path / "runs" / experiment_record["dataset"]["manifest_path"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert {sample["split"] for sample in manifest["samples"]} == {"train", "validation"}
    assert experiment_record["checkpoints"]["best"]["epoch"] == 1
    assert experiment_record["checkpoints"]["latest"]["supports_resume"] is True
    assert get_experiment(tmp_path / "runs", experiment_record["id"])["metrics"] == report
    assert list_experiments(tmp_path / "runs")[0]["id"] == experiment_record["id"]
    assert list_model_assets(tmp_path / "runs")[0]["experiment_id"] == experiment_record["id"]
    assert "INFO uatf.training.trainer: Epoch 1/1" in log_text
    assert "DEBUG uatf.cli: Effective configuration:" in log_text


    resume_checkpoint = torch.load(
        experiment_dir / "last_model.pth", map_location="cpu", weights_only=True
    )
    resume_checkpoint["best_metric"] = -1e6
    resume_path = experiment_dir / "resume-without-improvement.pth"
    torch.save(resume_checkpoint, resume_path)
    previous_best = torch.load(
        experiment_dir / "best_model.pth", map_location="cpu", weights_only=True
    )

    resumed_dir = run_training(
        {
            "dataset": {
                "path": str(dataset_dir),
                "validation_fraction": 0.25,
                "image_size": 16,
            },
            "model": {"name": "test_tiny", "pretrained": False},
            "training": {
                "epochs": 2,
                "batch_size": 2,
                "learning_rate": 0.08,
                "device": "cpu",
                "optimizer": {"name": "sgd", "momentum": 0.6},
                "resume_from": str(resume_path),
            },
            "output": {
                "directory": str(tmp_path / "runs"),
                "save_predictions": False,
                "save_plots": False,
            },
        }
    )
    resumed_report = json.loads((resumed_dir / "metrics.json").read_text(encoding="utf-8"))
    resumed_config = yaml.safe_load((resumed_dir / "config.yaml").read_text(encoding="utf-8"))
    resumed_best = torch.load(
        resumed_dir / "best_model.pth", map_location="cpu", weights_only=True
    )
    assert resumed_report["training"]["epochs_completed"] == 2
    assert [row["epoch"] for row in resumed_report["training"]["history"]] == [1.0, 2.0]
    assert resumed_config["training"]["learning_rate"] == 0.02
    assert all(
        torch.equal(previous_best["model_state_dict"][name], tensor)
        for name, tensor in resumed_best["model_state_dict"].items()
    )

    with pytest.raises(
        ValueError,
        match=r"training.epochs \(1\).*resumed checkpoint epoch \(2\)",
    ):
        run_training(
            {
                "dataset": {
                    "path": str(dataset_dir),
                    "image_size": 16,
                    "validation_fraction": 0.25,
                },
                "model": {"name": "test_tiny", "pretrained": False},
                "training": {
                    "epochs": 1,
                    "batch_size": 2,
                    "device": "cpu",
                    "optimizer": {"name": "sgd", "momentum": 0.6},
                    "resume_from": str(resumed_dir / "last_model.pth"),
                },
                "output": {"directory": str(tmp_path / "runs")},
            }
        )


def test_folder_dataset_with_explicit_train_and_validation_directories_is_recorded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(MODEL_REGISTRY, "test_tiny", _build_tiny_model)
    train_dir = tmp_path / "train"
    validation_dir = tmp_path / "validation"
    for split_dir in (train_dir, validation_dir):
        for class_name, color in (("blue", (0, 0, 220)), ("red", (220, 0, 0))):
            class_dir = split_dir / class_name
            class_dir.mkdir(parents=True)
            Image.new("RGB", (16, 16), color).save(class_dir / "sample.png")

    experiment_dir = run_training(
        {
            "dataset": {
                "train_dir": str(train_dir),
                "validation_dir": str(validation_dir),
                "image_size": 16,
            },
            "model": {"name": "test_tiny", "pretrained": False},
            "training": {"epochs": 1, "batch_size": 2, "device": "cpu"},
            "output": {
                "directory": str(tmp_path / "explicit-split-runs"),
                "save_predictions": False,
                "save_plots": False,
            },
        }
    )

    record = json.loads((experiment_dir / "experiment.json").read_text(encoding="utf-8"))
    assert record["status"] == "Completed"
    assert record["dataset"]["train_samples"] == 2
    assert record["dataset"]["validation_samples"] == 2
    assert record["dataset"]["validation_path"] == str(validation_dir.resolve())


def test_python_api_stop_persists_stopped_status_and_last_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(MODEL_REGISTRY, "test_tiny", _build_tiny_model)
    dataset_dir = tmp_path / "stop-images"
    for class_name, color in (("blue", (0, 0, 220)), ("red", (220, 0, 0))):
        class_dir = dataset_dir / class_name
        class_dir.mkdir(parents=True)
        for index in range(4):
            Image.new("RGB", (16, 16), color).save(class_dir / f"{index}.png")

    configuration = {
        "dataset": {"path": str(dataset_dir), "validation_fraction": 0.25, "image_size": 16},
        "model": {"name": "test_tiny", "pretrained": False},
        "training": {
            "epochs": 4,
            "batch_size": 2,
            "learning_rate": 0.01,
            "device": "cpu",
            "num_workers": 0,
            "optimizer": {"name": "sgd", "momentum": 0.4},
            "scheduler": {"enabled": True, "step_size": 1, "gamma": 0.5},
        },
        "output": {"directory": str(tmp_path / "stop-runs"), "save_plots": False},
    }
    stop_event = threading.Event()

    def request_stop_after_first_epoch(event: object) -> None:
        if getattr(event, "event_type") == "epoch_completed":
            stop_event.set()

    experiment_dir = run_training(
        configuration,
        task_id="api-stop",
        stop_requested=stop_event.is_set,
        event_callback=request_stop_after_first_epoch,
    )

    task = get_latest_task(tmp_path / "stop-runs")
    checkpoint = torch.load(
        experiment_dir / "last_model.pth", map_location="cpu", weights_only=True
    )
    assert task["status"] == "Stopped"
    assert task["task_id"] == "api-stop"
    assert task["epoch"] == 1
    assert checkpoint["epoch"] == 1
    assert not (experiment_dir / "metrics.json").exists()
    experiment_record = json.loads(
        (experiment_dir / "experiment.json").read_text(encoding="utf-8")
    )
    assert experiment_record["status"] == "Stopped"
    assert experiment_record["training"]["epochs_completed"] == 1
    assert experiment_record["checkpoints"]["latest"]["epoch"] == 1
    assert experiment_record["model_asset_id"]


def test_preflight_failure_persists_an_experiment_record(tmp_path: Path) -> None:
    empty_dataset = tmp_path / "empty"
    empty_dataset.mkdir()
    output_directory = tmp_path / "runs"
    configuration = {
        "task": {"type": "classification"},
        "dataset": {"path": str(empty_dataset), "validation_fraction": 0.2},
        "model": {"name": "resnet50", "pretrained": False},
        "training": {"epochs": 1, "batch_size": 1, "device": "cpu"},
        "output": {"directory": str(output_directory), "save_plots": False},
    }

    with pytest.raises(ValueError, match="must contain one subdirectory per class"):
        run_training(configuration)

    experiment_dir = output_directory / "experiment_001"
    record = json.loads((experiment_dir / "experiment.json").read_text(encoding="utf-8"))
    assert record["status"] == "Failed"
    assert record["failure"]["type"] == "DatasetValidationError"
    assert "one subdirectory per class" in record["failure"]["message"]
    assert (experiment_dir / "logs" / "train.log").is_file()
    assert get_task(output_directory, get_latest_task(output_directory)["task_id"])["status"] == "Failed"


def test_gui_stop_stops_cli_subprocess_and_checkpoint_can_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_dir = tmp_path / "subprocess-runs"
    dataset_dir = tmp_path / "subprocess-images"
    for class_name, color in (("blue", (0, 0, 220)), ("red", (220, 0, 0))):
        class_dir = dataset_dir / class_name
        class_dir.mkdir(parents=True)
        for index in range(4):
            Image.new("RGB", (16, 16), color).save(class_dir / f"{index}.png")

    task_id = "gui-subprocess-stop"
    monkeypatch.setattr(gui_app, "OUTPUT_ROOT", output_dir)
    monkeypatch.setitem(MODEL_REGISTRY, "test_subprocess_tiny", _build_subprocess_tiny_model)
    config_path = tmp_path / "subprocess-config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "dataset": {
                    "path": str(dataset_dir),
                    "validation_fraction": 0.25,
                    "image_size": 16,
                },
                "model": {"name": "test_subprocess_tiny", "pretrained": False},
                "training": {
                    "epochs": 20,
                    "batch_size": 2,
                    "learning_rate": 0.01,
                    "seed": 7,
                    "device": "cpu",
                    "num_workers": 0,
                    "optimizer": {"name": "sgd", "momentum": 0.4},
                    "scheduler": {"enabled": True, "step_size": 1, "gamma": 0.5},
                },
                "output": {"directory": str(output_dir), "save_plots": False},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    child_script = """
import sys
import time
from torch import nn
from uatf.models.factory import MODEL_REGISTRY

class SlowTinyModel(nn.Module):
    def __init__(self, output_count):
        super().__init__()
        self.network = nn.Sequential(nn.Flatten(), nn.Linear(3 * 16 * 16, output_count))

    def forward(self, images):
        time.sleep(0.08)
        return self.network(images)

MODEL_REGISTRY['test_subprocess_tiny'] = lambda output_count, pretrained: SlowTinyModel(output_count)
from uatf.cli import main
raise SystemExit(main(sys.argv[1:]))
"""
    process = subprocess.Popen(
        [
            sys.executable,
            "-c",
            child_script,
            "--config",
            str(config_path),
            "--task-id",
            task_id,
        ],
        cwd=Path(__file__).resolve().parents[2],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    try:
        deadline = time.monotonic() + 60
        task_record = None
        checkpoint_path = output_dir / "experiment_001" / "last_model.pth"
        while time.monotonic() < deadline:
            if process.poll() is not None:
                output, _ = process.communicate()
                pytest.fail(f"CLI subprocess exited before a checkpoint was saved:\n{output}")
            try:
                task_record = get_task(output_dir, task_id)
            except FileNotFoundError:
                time.sleep(0.05)
                continue
            if checkpoint_path.is_file():
                break
            time.sleep(0.05)
        else:
            pytest.fail("CLI subprocess did not save its first complete checkpoint in time.")

        assert task_record is not None
        assert task_record["status"] in {"Running", "Validating"}
        gui_message = gui_app.request_training_stop(task_id)
        assert "正在停止" in gui_message
        return_code = process.wait(timeout=60)
        output, _ = process.communicate()
        assert return_code == 130, output

        stopped_task = get_task(output_dir, task_id)
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        stopped_epoch = int(checkpoint["epoch"])
        assert stopped_task["status"] == "Stopped"
        assert stopped_task["epoch"] == stopped_epoch
        assert 1 <= stopped_epoch < 20
        assert checkpoint["optimizer_state_dict"]
        assert checkpoint["scheduler_state_dict"]

        probe = TaskManager(output_dir, "lock-release-probe")
        probe.acquire()
        probe.update("Stopped", stage="stopped")
        probe.release()

        resumed_directory = run_training(
            {
                "dataset": {
                    "path": str(dataset_dir),
                    "validation_fraction": 0.25,
                    "image_size": 16,
                },
                "model": {"name": "test_subprocess_tiny", "pretrained": False},
                "training": {
                    "epochs": stopped_epoch + 1,
                    "batch_size": 2,
                    "learning_rate": 0.01,
                    "seed": 7,
                    "device": "cpu",
                    "num_workers": 0,
                    "optimizer": {"name": "sgd", "momentum": 0.4},
                    "scheduler": {"enabled": True, "step_size": 1, "gamma": 0.5},
                    "resume_from": str(checkpoint_path),
                },
                "output": {"directory": str(output_dir), "save_plots": False},
            },
            task_id="resume-after-subprocess-stop",
        )
        resumed_report = json.loads(
            (resumed_directory / "metrics.json").read_text(encoding="utf-8")
        )
        assert [
            int(row["epoch"]) for row in resumed_report["training"]["history"]
        ] == list(range(1, stopped_epoch + 2))
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        if process.stdout is not None:
            process.stdout.close()


def test_invalid_configuration_is_kept_as_a_failed_task_record(tmp_path: Path) -> None:
    output_dir = tmp_path / "failed-runs"
    with pytest.raises(ValueError):
        run_training(
            {
                "task": {"type": "unsupported"},
                "output": {"directory": str(output_dir)},
            },
            task_id="bad-config",
        )

    task = get_latest_task(output_dir)
    assert task["status"] == "Failed"
    assert task["task_id"] == "bad-config"
    assert task["error_summary"]
