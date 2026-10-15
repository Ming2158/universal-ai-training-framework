"""Configuration-driven Python API for complete UATF training runs."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any
from uuid import uuid4

from uatf.config import ConfigurationError, load_config, resolve_config
from uatf.evaluation import evaluate_model_asset
from uatf.experiment import run_experiment
from uatf.experiments import (
    compare_experiments,
    export_model_asset,
    get_experiment,
    list_experiments,
    list_model_assets,
    load_exported_model_asset,
    load_model_asset,
    load_model_asset_bundle,
)
from uatf.inference import predict_model_asset_directory, predict_model_asset_image
from uatf.tasks import (
    TERMINAL_STATUSES,
    TaskManager,
    get_latest_task,
    get_task,
    request_task_stop,
)
from uatf.training import TrainingEvent


def run_training(
    configuration: Mapping[str, Any] | str | Path,
    *,
    overrides: Mapping[str, Any] | None = None,
    source_config_path: str | Path | None = None,
    event_callback: Callable[[TrainingEvent], None] | None = None,
    task_id: str | None = None,
    stop_requested: Callable[[], bool] | None = None,
) -> Path:
    """Validate a mapping or YAML file and run one complete training experiment.

    Explicit ``overrides`` take precedence over supplied configuration values;
    framework defaults are applied after both sources have been combined.
    """
    if isinstance(configuration, (str, Path)):
        config_path = Path(configuration).expanduser().resolve()
        supplied = load_config(config_path)
        if source_config_path is not None:
            raise ConfigurationError(
                "source_config_path cannot be set when configuration is a file path."
            )
        source_path: Path | None = config_path
    elif isinstance(configuration, Mapping):
        supplied = dict(configuration)
        source_path = (
            Path(source_config_path).expanduser().resolve()
            if source_config_path is not None
            else None
        )
    else:
        raise ConfigurationError("configuration must be a mapping or YAML file path.")

    output_overrides = overrides.get("output", {}) if isinstance(overrides, Mapping) else {}
    configured_output = supplied.get("output", {})
    output_directory = (
        output_overrides.get("directory")
        if isinstance(output_overrides, Mapping) and output_overrides.get("directory")
        else configured_output.get("directory")
        if isinstance(configured_output, Mapping)
        else None
    )
    if not isinstance(output_directory, str) or not output_directory.strip():
        output_directory = "outputs"
    candidate = Path(output_directory).expanduser()
    if candidate.exists() and not candidate.is_dir():
        candidate = Path("outputs")
    manager = TaskManager(candidate, task_id or uuid4().hex)
    manager.acquire()
    try:
        effective = resolve_config(supplied, overrides)
        manager.mark_preparing()

        def update_status(status: str, epoch: int | None, total_epochs: int | None) -> None:
            if status in {"Running", "Validating"}:
                manager.set_stage(
                    "validation" if status == "Validating" else "training",
                    epoch=epoch,
                    total_epochs=total_epochs,
                )
            else:
                manager.update(
                    status,  # type: ignore[arg-type]
                    stage=status.lower(),
                    epoch=epoch,
                    total_epochs=total_epochs,
                )

        def check_stop() -> bool:
            return manager.stop_requested() or bool(stop_requested and stop_requested())

        experiment_directory = run_experiment(
            effective,
            source_path,
            source_configuration=supplied,
            configuration_overrides=overrides,
            event_callback=event_callback,
            stop_requested=check_stop,
            task_status_callback=update_status,
            task_directory_callback=manager.set_experiment_directory,
        )
        current = manager.update()
        if current.get("status") not in TERMINAL_STATUSES:
            manager.update("Completed", stage="completed")
        return experiment_directory
    except BaseException as error:
        try:
            current = manager.update()
            if current.get("status") not in TERMINAL_STATUSES:
                manager.update("Failed", error_summary=str(error), stage="failed")
        except (OSError, ValueError):
            pass
        raise
    finally:
        manager.release()


__all__ = [
    "compare_experiments",
    "export_model_asset",
    "evaluate_model_asset",
    "get_experiment",
    "get_latest_task",
    "get_task",
    "list_experiments",
    "list_model_assets",
    "load_model_asset",
    "load_model_asset_bundle",
    "load_exported_model_asset",
    "predict_model_asset_directory",
    "predict_model_asset_image",
    "request_task_stop",
    "run_training",
]
