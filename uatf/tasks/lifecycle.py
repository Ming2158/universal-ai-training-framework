"""Cross-process task status, single-run locking, and cooperative stop requests."""

from __future__ import annotations

import json
import os
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from errno import EACCES, EAGAIN
from pathlib import Path
from typing import Any, Literal

TaskStatus = Literal[
    "Pending",
    "Preparing",
    "Running",
    "Validating",
    "Stopping",
    "Stopped",
    "Completed",
    "Failed",
]

ACTIVE_STATUSES = {"Pending", "Preparing", "Running", "Validating", "Stopping"}
TERMINAL_STATUSES = {"Stopped", "Completed", "Failed"}
_TRANSITIONS: dict[str, set[str]] = {
    "Pending": {"Preparing", "Stopping", "Stopped", "Failed"},
    "Preparing": {"Running", "Stopping", "Stopped", "Completed", "Failed"},
    "Running": {"Validating", "Stopping", "Stopped", "Completed", "Failed"},
    "Validating": {"Running", "Stopping", "Stopped", "Completed", "Failed"},
    "Stopping": {"Stopped", "Completed", "Failed"},
    "Stopped": set(),
    "Completed": set(),
    "Failed": set(),
}


class TaskAlreadyRunning(RuntimeError):
    """Raised when another training task owns the output directory lock."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _read_json(path: Path) -> dict[str, Any]:
    for attempt in range(3):
        try:
            content = path.read_text(encoding="utf-8")
            break
        except PermissionError:
            if attempt == 2:
                raise
            time.sleep(0.02 * (attempt + 1))
    value = json.loads(content)
    if not isinstance(value, dict):
        raise ValueError(f"Invalid task record: {path}")
    return value


def _pid_is_alive(pid: Any) -> bool:
    if isinstance(pid, bool) or not isinstance(pid, int) or pid < 1:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError as error:
        if getattr(error, "winerror", None) == 87:
            return False
        return True
    return True


def _os_lock_is_held(lock_path: Path) -> bool | None:
    try:
        lock_file = lock_path.open("r+b")
    except FileNotFoundError:
        return False
    except OSError:
        return None

    acquired = False
    try:
        lock_file.seek(0)
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(lock_file.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            acquired = True
            return False
        except OSError as error:
            if os.name == "nt" or isinstance(error, BlockingIOError):
                return True
            if error.errno in {EACCES, EAGAIN}:
                return True
            return None
    finally:
        if acquired:
            try:
                if os.name == "nt":
                    import msvcrt

                    lock_file.seek(0)
                    msvcrt.locking(lock_file.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
            finally:
                lock_file.close()
        else:
            lock_file.close()


def _task_lock_is_held(output_directory: str | Path, task_id: str) -> bool | None:
    """Check the task-specific lock instead of trusting a reusable PID."""
    task_lock = (
        Path(output_directory).expanduser().resolve()
        / ".uatf"
        / "tasks"
        / f"{task_id}.running.lock"
    )
    return _os_lock_is_held(task_lock)


@contextmanager
def _record_lock(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as lock_file:
        lock_file.seek(0, os.SEEK_END)
        if lock_file.tell() == 0:
            lock_file.write(b"\0")
            lock_file.flush()
        lock_file.seek(0)
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(lock_file.fileno(), msvcrt.LK_LOCK, 1)

            def unlock() -> None:
                msvcrt.locking(lock_file.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)

            def unlock() -> None:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
        try:
            yield
        finally:
            unlock()


class TaskManager:
    """Persist one task and hold output-root and task-specific process locks."""

    def __init__(
        self,
        output_directory: str | Path,
        task_id: str | None = None,
        *,
        process_id: int | None = None,
    ) -> None:
        self.output_directory = Path(output_directory).expanduser().resolve()
        self.task_id = task_id or uuid.uuid4().hex
        if not self.task_id or any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for character in self.task_id):
            raise ValueError("Task id may contain only letters, numbers, '-' and '_'.")
        self.process_id = process_id or os.getpid()
        self.metadata_directory = self.output_directory / ".uatf"
        self.tasks_directory = self.metadata_directory / "tasks"
        self.record_path = self.tasks_directory / f"{self.task_id}.json"
        self.stop_path = self.tasks_directory / f"{self.task_id}.stop"
        self.record_lock_path = self.tasks_directory / f"{self.task_id}.lock"
        self.running_lock_path = self.tasks_directory / f"{self.task_id}.running.lock"
        self.lock_path = self.metadata_directory / "active-task.lock"
        self._owns_lock = False
        self._owns_running_lock = False
        self._lock_file: Any = None
        self._running_lock_file: Any = None

    def acquire(self) -> None:
        self.tasks_directory.mkdir(parents=True, exist_ok=True)
        lock_record = {"task_id": self.task_id, "process_id": self.process_id}
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        lock_file = self.lock_path.open("a+b")
        try:
            lock_file.seek(0, os.SEEK_END)
            if lock_file.tell() == 0:
                lock_file.write(b"\0")
                lock_file.flush()
            lock_file.seek(0)
            if os.name == "nt":
                import msvcrt

                try:
                    msvcrt.locking(lock_file.fileno(), msvcrt.LK_NBLCK, 1)
                except OSError as error:
                    raise self._already_running(error) from error
            else:
                import fcntl

                try:
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError as error:
                    raise self._already_running(error) from error
            self._lock_file = lock_file
            self._owns_lock = True
            lock_file.seek(0)
            lock_file.truncate()
            lock_file.write(json.dumps(lock_record).encode("utf-8") + b"\n")
            lock_file.flush()

            running_lock_file = self.running_lock_path.open("a+b")
            try:
                running_lock_file.seek(0, os.SEEK_END)
                if running_lock_file.tell() == 0:
                    running_lock_file.write(b"\0")
                    running_lock_file.flush()
                running_lock_file.seek(0)
                try:
                    if os.name == "nt":
                        import msvcrt

                        msvcrt.locking(running_lock_file.fileno(), msvcrt.LK_NBLCK, 1)
                    else:
                        import fcntl

                        fcntl.flock(
                            running_lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB
                        )
                except OSError as error:
                    raise self._already_running(error) from error
            except BaseException:
                running_lock_file.close()
                raise
            self._running_lock_file = running_lock_file
            self._owns_running_lock = True
        except BaseException:
            if self._owns_lock:
                self.release()
            else:
                lock_file.close()
            raise

        try:
            if self.record_path.exists():
                raise FileExistsError(f"Task id already exists: {self.task_id}")
            now = _now()
            _atomic_json(
                self.record_path,
                {
                    "task_id": self.task_id,
                    "status": "Pending",
                    "created_at": now,
                    "updated_at": now,
                    "started_at": None,
                    "finished_at": None,
                    "process_id": self.process_id,
                    "experiment_id": None,
                    "result_directory": None,
                    "stage": None,
                    "epoch": None,
                    "total_epochs": None,
                    "error_summary": None,
                    "stop_requested": False,
                },
            )
            _atomic_json(
                self.metadata_directory / "latest-task.json", {"task_id": self.task_id}
            )
        except BaseException:
            self.release()
            raise

    def release(self) -> None:
        if self._owns_running_lock:
            self._owns_running_lock = False
            running_lock_file, self._running_lock_file = self._running_lock_file, None
            if running_lock_file is not None:
                try:
                    if os.name == "nt":
                        import msvcrt

                        running_lock_file.seek(0)
                        msvcrt.locking(
                            running_lock_file.fileno(), msvcrt.LK_UNLCK, 1
                        )
                    else:
                        import fcntl

                        fcntl.flock(running_lock_file.fileno(), fcntl.LOCK_UN)
                finally:
                    running_lock_file.close()
        if not self._owns_lock:
            return
        self._owns_lock = False
        lock_file, self._lock_file = self._lock_file, None
        if lock_file is None:
            return
        try:
            if os.name == "nt":
                import msvcrt

                lock_file.seek(0)
                msvcrt.locking(lock_file.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
        finally:
            lock_file.close()

    def _already_running(self, error: OSError) -> TaskAlreadyRunning:
        try:
            current = _read_json(self.lock_path)
            detail = f"Training task {current.get('task_id', 'unknown')} is already active"
        except (OSError, json.JSONDecodeError, ValueError):
            detail = "A training task is already active"
        return TaskAlreadyRunning(f"{detail} for output directory {self.output_directory}.")

    def __enter__(self) -> TaskManager:
        self.acquire()
        return self

    def __exit__(self, *_: object) -> None:
        self.release()

    def update(
        self,
        status: TaskStatus | None = None,
        *,
        experiment_directory: str | Path | None = None,
        stage: str | None = None,
        epoch: int | None = None,
        total_epochs: int | None = None,
        error_summary: str | None = None,
    ) -> dict[str, Any]:
        with _record_lock(self.record_lock_path):
            record = _read_json(self.record_path)
            current_status = record.get("status")
            if status is not None and status != current_status:
                if status not in _TRANSITIONS.get(current_status, set()):
                    raise ValueError(
                        f"Invalid task status transition: {current_status} -> {status}."
                    )
                record["status"] = status
                if status == "Preparing" and record.get("started_at") is None:
                    record["started_at"] = _now()
                if status in TERMINAL_STATUSES:
                    record["finished_at"] = _now()
            if experiment_directory is not None:
                result_directory = str(Path(experiment_directory).resolve())
                record["experiment_id"] = Path(result_directory).name
                record["result_directory"] = result_directory
            if stage is not None:
                record["stage"] = stage
            if epoch is not None:
                record["epoch"] = epoch
            if total_epochs is not None:
                record["total_epochs"] = total_epochs
            if error_summary is not None:
                record["error_summary"] = error_summary.strip()[:1000] or None
            record["updated_at"] = _now()
            _atomic_json(self.record_path, record)
            return record

    def set_stage(self, stage: str, *, epoch: int | None = None, total_epochs: int | None = None) -> None:
        status: TaskStatus = "Validating" if stage == "validation" else "Running"
        with _record_lock(self.record_lock_path):
            record = _read_json(self.record_path)
            if record.get("status") == "Stopping":
                return
            current_status = record.get("status")
            if status != current_status:
                if status not in _TRANSITIONS.get(current_status, set()):
                    raise ValueError(
                        f"Invalid task status transition: {current_status} -> {status}."
                    )
                record["status"] = status
            record["stage"] = stage
            if epoch is not None:
                record["epoch"] = epoch
            if total_epochs is not None:
                record["total_epochs"] = total_epochs
            record["updated_at"] = _now()
            _atomic_json(self.record_path, record)

    def mark_preparing(self) -> dict[str, Any]:
        with _record_lock(self.record_lock_path):
            record = _read_json(self.record_path)
            if record.get("status") == "Stopping":
                return record
            if record.get("status") == "Pending":
                record["status"] = "Preparing"
                record["started_at"] = _now()
            record["updated_at"] = _now()
            _atomic_json(self.record_path, record)
            return record

    def set_experiment_directory(self, directory: str | Path) -> None:
        self.update(experiment_directory=directory)

    def stop_requested(self) -> bool:
        return self.stop_path.is_file()

    def request_stop(self) -> dict[str, Any]:
        with _record_lock(self.record_lock_path):
            record = _read_json(self.record_path)
            if record.get("status") in TERMINAL_STATUSES:
                return record
            self.tasks_directory.mkdir(parents=True, exist_ok=True)
            self.stop_path.write_text(_now() + "\n", encoding="utf-8")
            record["stop_requested"] = True
            record["updated_at"] = _now()
            current_status = record.get("status")
            if current_status in ACTIVE_STATUSES and current_status != "Stopping":
                if "Stopping" not in _TRANSITIONS[current_status]:
                    raise ValueError(f"Task in {current_status} state cannot be stopped.")
                record["status"] = "Stopping"
            _atomic_json(self.record_path, record)
            return record


def _record_for_directory(output_directory: str | Path, task_id: str) -> tuple[Path, Path]:
    if not task_id or any(
        character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
        for character in task_id
    ):
        raise ValueError("Task id may contain only letters, numbers, '-' and '_'.")
    root = Path(output_directory).expanduser().resolve() / ".uatf"
    return root, root / "tasks" / f"{task_id}.json"


def get_task(
    output_directory: str | Path,
    task_id: str,
    *,
    reconcile_interruption: bool = True,
) -> dict[str, Any]:
    root, path = _record_for_directory(output_directory, task_id)
    record = _read_json(path)
    if reconcile_interruption and record.get("status") in ACTIVE_STATUSES:
        lock_held = _task_lock_is_held(output_directory, task_id)
        if lock_held is True:
            process_active = True
        elif lock_held is False:
            legacy_lock_held = _os_lock_is_held(root / "active-task.lock")
            try:
                latest_task_id = _read_json(root / "latest-task.json").get("task_id")
            except (OSError, json.JSONDecodeError, ValueError):
                latest_task_id = None
            if legacy_lock_held is True:
                process_active = latest_task_id == task_id
            elif legacy_lock_held is False:
                process_active = False
            else:
                process_active = _pid_is_alive(record.get("process_id"))
        else:
            process_active = _pid_is_alive(record.get("process_id"))
    else:
        process_active = True
    if reconcile_interruption and record.get("status") in ACTIVE_STATUSES and not process_active:
        manager = TaskManager(output_directory, task_id, process_id=record.get("process_id"))
        record = manager.update(
            "Failed",
            error_summary="训练进程已退出，任务未能正常完成；请检查日志和最近检查点。",
        )
    return record


def get_latest_task(output_directory: str | Path) -> dict[str, Any] | None:
    root = Path(output_directory).expanduser().resolve() / ".uatf"
    pointer = root / "latest-task.json"
    try:
        task_id = _read_json(pointer).get("task_id")
    except (OSError, json.JSONDecodeError, ValueError):
        return None
    if not isinstance(task_id, str):
        return None
    try:
        return get_task(output_directory, task_id)
    except (OSError, json.JSONDecodeError, ValueError):
        return None


def request_task_stop(output_directory: str | Path, task_id: str) -> dict[str, Any]:
    manager = TaskManager(output_directory, task_id)
    if manager.record_path.is_file():
        return manager.request_stop()
    manager.tasks_directory.mkdir(parents=True, exist_ok=True)
    manager.stop_path.write_text(_now() + "\n", encoding="utf-8")
    return {
        "task_id": task_id,
        "status": "Stopping",
        "stop_requested": True,
    }
