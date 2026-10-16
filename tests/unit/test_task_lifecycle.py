from __future__ import annotations

import os
from pathlib import Path

import pytest

from uatf.tasks import (
    TaskAlreadyRunning,
    TaskManager,
    get_latest_task,
    get_task,
    request_task_stop,
)


def test_task_status_transitions_are_persisted_with_lifecycle_metadata(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "runs"
    experiment_dir = output_dir / "experiment_001"
    manager = TaskManager(output_dir, "task-1")
    manager.acquire()
    try:
        manager.update("Preparing")
        manager.set_experiment_directory(experiment_dir)
        manager.set_stage("training", epoch=1, total_epochs=3)
        manager.set_stage("validation", epoch=1, total_epochs=3)
        manager.set_stage("training", epoch=2, total_epochs=3)
        completed = manager.update("Completed", stage="completed")
    finally:
        manager.release()

    assert completed["status"] == "Completed"
    assert completed["task_id"] == "task-1"
    assert completed["experiment_id"] == "experiment_001"
    assert completed["result_directory"] == str(experiment_dir.resolve())
    assert completed["process_id"] > 0
    assert completed["started_at"]
    assert completed["finished_at"]
    assert completed["updated_at"] >= completed["created_at"]
    assert get_task(output_dir, "task-1")["status"] == "Completed"
    assert get_latest_task(output_dir)["task_id"] == "task-1"

    with pytest.raises(ValueError, match="Invalid task status transition"):
        manager.update("Running")


def test_task_json_read_retries_a_transient_permission_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager = TaskManager(tmp_path / "runs", "transient-read")
    manager.acquire()
    manager.update("Preparing")
    manager.release()

    original_read_text = Path.read_text
    failed_once = False

    def read_with_one_temporary_denial(path: Path, *args, **kwargs) -> str:
        nonlocal failed_once
        if path == manager.record_path and not failed_once:
            failed_once = True
            raise PermissionError("temporary file share conflict")
        return original_read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", read_with_one_temporary_denial)

    assert (
        get_task(tmp_path / "runs", "transient-read", reconcile_interruption=False)["status"]
        == "Preparing"
    )
    assert failed_once


def test_only_one_task_can_hold_an_output_root_lock(tmp_path: Path) -> None:
    first = TaskManager(tmp_path / "runs", "first")
    second = TaskManager(tmp_path / "runs", "second")
    first.acquire()
    try:
        with pytest.raises(TaskAlreadyRunning, match="already active"):
            second.acquire()
    finally:
        first.release()

    second.acquire()
    second.release()


def test_stop_request_is_persisted_and_transitions_to_stopped(tmp_path: Path) -> None:
    output_dir = tmp_path / "runs"
    manager = TaskManager(output_dir, "stop-me")
    manager.acquire()
    manager.update("Preparing")
    manager.set_stage("training", epoch=1, total_epochs=5)

    requested = manager.request_stop()

    assert requested["status"] == "Stopping"
    assert requested["stop_requested"] is True
    assert manager.stop_requested() is True
    assert manager.update("Stopped", stage="stopped")["status"] == "Stopped"
    manager.release()

    with pytest.raises(ValueError, match="Invalid task status transition"):
        manager.update("Running")


def test_stop_can_arrive_before_the_cli_creates_its_record(tmp_path: Path) -> None:
    output_dir = tmp_path / "runs"
    requested = request_task_stop(output_dir, "not-attached-yet")
    manager = TaskManager(output_dir, "not-attached-yet")
    manager.acquire()

    assert requested["status"] == "Stopping"
    assert manager.stop_requested() is True
    manager.update("Preparing")
    manager.update("Stopped")
    manager.release()


def test_preparing_transition_does_not_overwrite_a_concurrent_stop(tmp_path: Path) -> None:
    output_dir = tmp_path / "runs"
    manager = TaskManager(output_dir, "racing-stop")
    manager.acquire()
    manager.request_stop()

    record = manager.mark_preparing()

    assert record["status"] == "Stopping"
    assert manager.stop_requested() is True
    manager.update("Stopped")
    manager.release()


def test_query_does_not_reconcile_an_active_task_while_its_lock_is_held(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "runs"
    manager = TaskManager(output_dir, "interrupted", process_id=2_000_000_000)
    manager.acquire()
    try:
        manager.update("Preparing")
        manager.set_stage("training", epoch=2, total_epochs=4)

        record = get_task(output_dir, "interrupted")
    finally:
        manager.release()

    assert record["status"] == "Running"


def test_query_marks_a_released_task_failed_even_when_its_pid_is_alive(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "runs"
    manager = TaskManager(output_dir, "interrupted", process_id=os.getpid())
    manager.acquire()
    manager.update("Preparing")
    manager.set_stage("training", epoch=2, total_epochs=4)
    manager.release()

    record = get_task(output_dir, "interrupted")

    assert record["status"] == "Failed"
    assert "进程已退出" in record["error_summary"]
