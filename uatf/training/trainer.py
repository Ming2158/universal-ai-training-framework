"""A dataset- and model-agnostic PyTorch training engine."""

from __future__ import annotations

import logging
import pickle
from collections.abc import Callable, Iterable, Mapping
from contextlib import nullcontext
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from math import isfinite
from pathlib import Path
from time import perf_counter
from typing import Any, Literal

import torch
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

logger = logging.getLogger(__name__)

Metric = Callable[[Tensor, Tensor], float | Tensor]
BatchLoader = Iterable[tuple[Tensor, Tensor]]
TrainingEventCallback = Callable[["TrainingEvent"], None]
TaskStatusCallback = Callable[[str, int | None, int | None], None]
_PROGRESS_EVENT_INTERVAL_SECONDS = 1.0


@dataclass(frozen=True)
class TrainingEvent:
    """Structured lifecycle data emitted by configuration-driven training."""

    event_type: Literal[
        "training_started",
        "training_progress",
        "epoch_completed",
        "training_completed",
        "training_failed",
        "training_stopping",
        "training_stopped",
    ]
    experiment_dir: str | None = None
    epoch: int | None = None
    total_epochs: int | None = None
    metrics: Mapping[str, float] | None = None
    message: str | None = None
    stage: Literal["training", "validation"] | None = None
    batch_index: int | None = None
    total_batches: int | None = None
    elapsed_seconds: float | None = None
    timestamp_utc: str | None = None


class TrainingEventEmitter:
    """Add timestamps and isolate optional observers from the training loop."""

    def __init__(self, callback: TrainingEventCallback | None) -> None:
        self._callback = callback
        self._active = callback is not None

    def __call__(self, event: TrainingEvent) -> None:
        if not self._active or self._callback is None:
            return
        if event.timestamp_utc is None:
            event = replace(event, timestamp_utc=datetime.now(timezone.utc).isoformat())
        try:
            self._callback(event)
        except Exception:
            self._active = False
            logger.warning("Training event observer failed; continuing without monitoring.", exc_info=True)


def build_optimizer(
    parameters: Iterable[nn.Parameter],
    configuration: Mapping[str, Any],
    *,
    learning_rate: float,
) -> Optimizer:
    """Build a supported optimizer from normalized training configuration."""
    if not isinstance(configuration, Mapping):
        raise ValueError("Training optimizer configuration must be a mapping.")
    name = configuration.get("name", "adam")
    weight_decay = configuration.get("weight_decay", 0.0)
    if name == "adam":
        return torch.optim.Adam(
            parameters,
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=tuple(configuration.get("betas", (0.9, 0.999))),
            eps=configuration.get("eps", 1e-8),
        )
    if name == "adamw":
        return torch.optim.AdamW(
            parameters,
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=tuple(configuration.get("betas", (0.9, 0.999))),
            eps=configuration.get("eps", 1e-8),
        )
    if name == "sgd":
        return torch.optim.SGD(
            parameters,
            lr=learning_rate,
            weight_decay=weight_decay,
            momentum=configuration.get("momentum", 0.0),
            nesterov=configuration.get("nesterov", False),
        )
    raise ValueError(
        f"Unsupported optimizer {name!r}; currently supported: adam, adamw, sgd."
    )


@dataclass
class EarlyStopping:
    """Stop after ``patience`` epochs without a meaningful monitor improvement."""

    patience: int
    min_delta: float = 0.0
    mode: Literal["min", "max"] = "min"
    best_metric: float | None = None
    bad_epochs: int = 0

    def __post_init__(self) -> None:
        if (
            isinstance(self.patience, bool)
            or not isinstance(self.patience, int)
            or self.patience < 1
        ):
            raise ValueError("Early stopping patience must be at least 1.")
        if not isfinite(self.min_delta) or self.min_delta < 0:
            raise ValueError("Early stopping min_delta must be finite and non-negative.")
        if self.mode not in {"min", "max"}:
            raise ValueError("Early stopping mode must be 'min' or 'max'.")

    def step(self, value: float) -> bool:
        """Record a monitor value and return whether training should stop."""
        improved = self.best_metric is None or (
            value < self.best_metric - self.min_delta
            if self.mode == "min"
            else value > self.best_metric + self.min_delta
        )
        if improved:
            self.best_metric = value
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1
        return self.bad_epochs >= self.patience

    def state_dict(self) -> dict[str, float | int | str | None]:
        return {
            "patience": self.patience,
            "min_delta": self.min_delta,
            "mode": self.mode,
            "best_metric": self.best_metric,
            "bad_epochs": self.bad_epochs,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if state.get("patience") != self.patience or state.get("mode") != self.mode:
            raise ValueError(
                "Early stopping settings do not match the checkpoint "
                "(patience and mode must be unchanged when resuming)."
            )
        if float(state.get("min_delta", 0.0)) != self.min_delta:
            raise ValueError("Early stopping min_delta does not match the checkpoint.")
        self.best_metric = state.get("best_metric")
        self.bad_epochs = int(state.get("bad_epochs", 0))


def build_scheduler(
    optimizer: Optimizer,
    configuration: Mapping[str, Any] | None,
    *,
    epochs: int | None = None,
) -> LRScheduler | None:
    """Build a supported epoch-based learning-rate scheduler when enabled."""
    if configuration is None:
        return None
    if not isinstance(configuration, Mapping):
        raise ValueError("Training scheduler configuration must be a mapping.")

    enabled = configuration.get("enabled", False)
    if not isinstance(enabled, bool):
        raise ValueError("Training scheduler 'enabled' must be true or false.")
    if not enabled:
        return None

    name = configuration.get("name", "step_lr")
    if name == "step_lr":
        step_size = configuration.get("step_size", 5)
        gamma = configuration.get("gamma", 0.1)
        if isinstance(step_size, bool) or not isinstance(step_size, int) or step_size < 1:
            raise ValueError("StepLR step_size must be a positive integer.")
        if (
            isinstance(gamma, bool)
            or not isinstance(gamma, (int, float))
            or not isfinite(gamma)
            or gamma <= 0
        ):
            raise ValueError("StepLR gamma must be finite and greater than 0.")
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    if name == "cosine":
        t_max = configuration.get("t_max", epochs or 10)
        eta_min = configuration.get("eta_min", 0.0)
        if isinstance(t_max, bool) or not isinstance(t_max, int) or t_max < 1:
            raise ValueError("CosineAnnealingLR t_max must be a positive integer.")
        if (
            isinstance(eta_min, bool)
            or not isinstance(eta_min, (int, float))
            or not isfinite(eta_min)
            or eta_min < 0
        ):
            raise ValueError("CosineAnnealingLR eta_min must be finite and non-negative.")
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=t_max, eta_min=float(eta_min)
        )
    raise ValueError("Unsupported scheduler; currently supported: step_lr, cosine.")


def build_early_stopping(configuration: Mapping[str, Any] | None) -> EarlyStopping | None:
    """Build optional early stopping from its small training config section."""
    if configuration is None:
        return None
    if not isinstance(configuration, Mapping):
        raise ValueError("Training early_stopping configuration must be a mapping.")
    enabled = configuration.get("enabled", False)
    if not isinstance(enabled, bool):
        raise ValueError("Training early_stopping 'enabled' must be true or false.")
    if not enabled:
        return None
    return EarlyStopping(
        patience=configuration.get("patience", 5),
        min_delta=configuration.get("min_delta", 0.0),
        mode=configuration.get("mode", "min"),
    )


class Trainer:
    """Train and validate any model that consumes batches of ``(inputs, targets)``."""

    def __init__(
        self,
        model: nn.Module,
        loss_function: nn.Module,
        optimizer: Optimizer,
        *,
        scheduler: LRScheduler | None = None,
        metrics: Mapping[str, Metric] | None = None,
        device: str | torch.device | None = None,
        checkpoint_dir: str | Path = "outputs",
        monitor: str = "val_loss",
        mode: Literal["min", "max"] = "min",
        early_stopping: EarlyStopping | None = None,
        amp: bool = False,
        gradient_accumulation_steps: int = 1,
        gradient_clip_norm: float | None = None,
        task_type: str | None = None,
        label_names: Iterable[str] | None = None,
        resume_metadata: Mapping[str, Any] | None = None,
        event_callback: TrainingEventCallback | None = None,
    ) -> None:
        if mode not in {"min", "max"}:
            raise ValueError("Trainer mode must be 'min' or 'max'.")
        if not monitor:
            raise ValueError("Trainer monitor cannot be empty.")
        if not isinstance(amp, bool):
            raise ValueError("Trainer amp must be true or false.")
        if (
            isinstance(gradient_accumulation_steps, bool)
            or not isinstance(gradient_accumulation_steps, int)
            or gradient_accumulation_steps < 1
        ):
            raise ValueError("gradient_accumulation_steps must be a positive integer.")
        if gradient_clip_norm is not None and (
            isinstance(gradient_clip_norm, bool)
            or not isinstance(gradient_clip_norm, (int, float))
            or not isfinite(gradient_clip_norm)
            or gradient_clip_norm <= 0
        ):
            raise ValueError("gradient_clip_norm must be finite and greater than 0 or None.")
        if early_stopping is not None and early_stopping.mode != mode:
            raise ValueError("Trainer mode and early stopping mode must match.")
        if metrics is not None:
            for name, metric in metrics.items():
                if not isinstance(name, str) or not name or name == "loss":
                    raise ValueError("Metric names must be non-empty strings; 'loss' is reserved.")
                if not callable(metric):
                    raise TypeError(f"Training metric {name!r} must be callable.")

        self.device = self._resolve_device(device)
        if amp and self.device.type != "cuda":
            raise ValueError("AMP requires a CUDA device; use amp=False for FP32 training.")

        self.model = model.to(self.device)
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.metrics = dict(metrics or {})
        self.checkpoint_dir = Path(checkpoint_dir)
        self.monitor = monitor
        self.mode = mode
        self.early_stopping = early_stopping
        self.task_type = task_type
        self.label_names = tuple(label_names) if label_names is not None else None
        self.resume_metadata = dict(resume_metadata) if resume_metadata is not None else None
        if self.label_names is not None and (
            any(not isinstance(name, str) or not name for name in self.label_names)
            or len(set(self.label_names)) != len(self.label_names)
        ):
            raise ValueError("Trainer label_names must contain unique, non-empty strings.")
        self.event_callback = event_callback
        self._event_emitter = TrainingEventEmitter(event_callback)
        self.amp = amp
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.gradient_clip_norm = (
            float(gradient_clip_norm) if gradient_clip_norm is not None else None
        )
        self.scaler = self._create_grad_scaler(amp)
        self.start_epoch = 1
        self._epoch = 0
        self._best_metric: float | None = None
        self._train_loader: BatchLoader | None = None
        self._pending_loader_generator_state: Tensor | None = None
        self._training_history: list[dict[str, float]] = []
        self.stopped = False

    @property
    def training_history(self) -> list[dict[str, float]]:
        """Return all known epoch records, including records restored from a checkpoint."""
        return [dict(record) for record in self._training_history]

    @staticmethod
    def _resolve_device(device: str | torch.device | None) -> torch.device:
        if device is None or device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        resolved = torch.device(device)
        if resolved.type == "cuda" and not torch.cuda.is_available():
            raise ValueError(
                "CUDA was requested but is unavailable in this PyTorch environment; "
                "use device='cpu' or install a CUDA-enabled PyTorch build."
            )
        return resolved

    @staticmethod
    def _create_grad_scaler(enabled: bool) -> Any:
        amp_api = getattr(torch, "amp", None)
        scaler_type = getattr(amp_api, "GradScaler", None)
        if scaler_type is not None:
            try:
                return scaler_type("cuda", enabled=enabled)
            except TypeError:
                pass
        return torch.cuda.amp.GradScaler(enabled=enabled)

    def train_epoch(
        self,
        train_loader: BatchLoader,
        *,
        epoch: int | None = None,
        total_epochs: int | None = None,
    ) -> dict[str, float]:
        """Run one optimization epoch and return mean loss and configured metrics."""
        self.model.train()
        return self._run_epoch(
            train_loader, training=True, epoch=epoch, total_epochs=total_epochs
        )

    def validate(
        self,
        validation_loader: BatchLoader,
        *,
        epoch: int | None = None,
        total_epochs: int | None = None,
    ) -> dict[str, float]:
        """Measure loss and metrics without changing model or optimizer state."""
        self.model.eval()
        with torch.no_grad():
            return self._run_epoch(
                validation_loader, training=False, epoch=epoch, total_epochs=total_epochs
            )

    def _run_epoch(
        self,
        loader: BatchLoader,
        *,
        training: bool,
        epoch: int | None = None,
        total_epochs: int | None = None,
    ) -> dict[str, float]:
        label = "training" if training else "validation"
        stage: Literal["training", "validation"] = "training" if training else "validation"
        totals = {name: 0.0 for name in self.metrics}
        total_loss = 0.0
        example_count = 0
        started = perf_counter()
        last_event_at = started
        last_event_batch = 0
        try:
            total_batches = len(loader)  # type: ignore[arg-type]
        except Exception:
            total_batches = None
        if total_batches is not None and total_batches < 1:
            total_batches = None
        pending_accumulation = 0
        pending_examples = 0
        if training:
            self.optimizer.zero_grad(set_to_none=True)

        for batch_index, batch in enumerate(loader, start=1):
            if not isinstance(batch, (tuple, list)) or len(batch) != 2:
                raise ValueError(
                    f"Invalid dataset batch {batch_index}: expected (inputs, targets)."
                )
            inputs, targets = batch
            if not isinstance(inputs, Tensor) or not isinstance(targets, Tensor):
                raise ValueError(
                    f"Invalid dataset batch {batch_index}: inputs and targets must be tensors."
                )
            if inputs.ndim == 0 or targets.ndim == 0:
                raise ValueError(
                    f"Invalid dataset batch {batch_index}: inputs and targets need a batch dimension."
                )
            batch_size = inputs.shape[0]
            if batch_size == 0 or targets.shape[0] != batch_size:
                raise ValueError(
                    f"Invalid dataset batch {batch_index}: input shape {tuple(inputs.shape)} "
                    f"and target shape {tuple(targets.shape)} have incompatible batch sizes."
                )

            try:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                with self._autocast_context():
                    try:
                        predictions = self.model(inputs)
                    except RuntimeError as error:
                        if any(
                            term in str(error).lower()
                            for term in ("shape", "size mismatch", "mat1", "channel", "dimension")
                        ):
                            raise ValueError(
                                f"Model shape mismatch for dataset batch {batch_index}: "
                                f"input shape {tuple(inputs.shape)}."
                            ) from error
                        raise
                    if not isinstance(predictions, Tensor):
                        raise TypeError("Training models must return a single prediction tensor.")
                    try:
                        loss = self.loss_function(predictions, targets)
                    except (RuntimeError, ValueError) as error:
                        raise ValueError(
                            f"Loss could not compare model output shape {tuple(predictions.shape)} "
                            f"with target shape {tuple(targets.shape)} in batch {batch_index}; "
                            "check the model output and dataset target formats."
                        ) from error

                if loss.numel() != 1:
                    raise ValueError("Loss function must return a single scalar value.")
                if not torch.isfinite(loss.detach()).all().item():
                    raise FloatingPointError(
                        f"NaN or infinite loss in {label} batch {batch_index}; "
                        "check the data, targets, learning rate, and model outputs."
                    )

                if training:
                    scaled_loss = loss * batch_size
                    self.scaler.scale(scaled_loss).backward()
                    pending_accumulation += 1
                    pending_examples += batch_size
                    batch_is_last = total_batches is not None and batch_index >= total_batches
                    if (
                        pending_accumulation >= self.gradient_accumulation_steps
                        or batch_is_last
                    ):
                        self._optimizer_step(pending_examples)
                        pending_accumulation = 0
                        pending_examples = 0

                detached_predictions = predictions.detach()
                detached_targets = targets.detach()
                batch_values: dict[str, float] = {}
                for name, metric in self.metrics.items():
                    metric_value = metric(detached_predictions, detached_targets)
                    if isinstance(metric_value, Tensor):
                        if metric_value.numel() != 1:
                            raise ValueError(f"Metric {name!r} must return one scalar value.")
                        metric_value = metric_value.item()
                    metric_value = float(metric_value)
                    if not isfinite(metric_value):
                        raise FloatingPointError(
                            f"Metric {name!r} returned NaN or infinity in batch {batch_index}."
                        )
                    batch_values[name] = metric_value

            except torch.cuda.OutOfMemoryError as error:
                raise RuntimeError(
                    f"CUDA ran out of memory in {label} batch {batch_index}. "
                    "Reduce batch_size, use a smaller model, or enable AMP on a supported GPU."
                ) from error
            except RuntimeError as error:
                if self.device.type == "cuda" and "out of memory" in str(error).lower():
                    raise RuntimeError(
                        f"CUDA ran out of memory in {label} batch {batch_index}. "
                        "Reduce batch_size, use a smaller model, or enable AMP on a supported GPU."
                    ) from error
                raise

            total_loss += float(loss.detach().item()) * batch_size
            for name, value in batch_values.items():
                totals[name] += value * batch_size
            example_count += batch_size

            now = perf_counter()
            is_last_batch = total_batches is not None and batch_index >= total_batches
            metric_prefix = "train" if training else "val"
            if (
                epoch is not None
                and total_epochs is not None
                and self.event_callback is not None
                and (
                    batch_index == 1
                    or is_last_batch
                    or now - last_event_at >= _PROGRESS_EVENT_INTERVAL_SECONDS
                )
            ):
                progress_metrics = {
                    f"{metric_prefix}_loss": total_loss / example_count,
                    **{
                        f"{metric_prefix}_{name}": value / example_count
                        for name, value in totals.items()
                    },
                }
                if training:
                    progress_metrics["learning_rate"] = float(
                        self.optimizer.param_groups[0]["lr"]
                    )
                self._event_emitter(
                    TrainingEvent(
                        event_type="training_progress",
                        experiment_dir=str(self.checkpoint_dir),
                        epoch=epoch,
                        total_epochs=total_epochs,
                        metrics=progress_metrics,
                        stage=stage,
                        batch_index=batch_index,
                        total_batches=total_batches,
                        elapsed_seconds=now - started,
                    )
                )
                last_event_at = now
                last_event_batch = batch_index

        if training and pending_accumulation:
            self._optimizer_step(pending_examples)

        if example_count == 0:
            raise ValueError(f"Invalid dataset: {label} DataLoader yielded no examples.")
        if (
            epoch is not None
            and total_epochs is not None
            and self.event_callback is not None
            and last_event_batch != batch_index
        ):
            metric_prefix = "train" if training else "val"
            progress_metrics = {
                f"{metric_prefix}_loss": total_loss / example_count,
                **{
                    f"{metric_prefix}_{name}": value / example_count
                    for name, value in totals.items()
                },
            }
            if training:
                progress_metrics["learning_rate"] = float(
                    self.optimizer.param_groups[0]["lr"]
                )
            self._event_emitter(
                TrainingEvent(
                    event_type="training_progress",
                    experiment_dir=str(self.checkpoint_dir),
                    epoch=epoch,
                    total_epochs=total_epochs,
                    metrics=progress_metrics,
                    stage=stage,
                    batch_index=batch_index,
                    total_batches=total_batches,
                    elapsed_seconds=perf_counter() - started,
                )
            )
        return {
            "loss": total_loss / example_count,
            **{name: value / example_count for name, value in totals.items()},
        }

    def _optimizer_step(self, accumulated_examples: int) -> None:
        if self.amp:
            self.scaler.unscale_(self.optimizer)
        for parameter in self.model.parameters():
            if parameter.grad is not None:
                parameter.grad.div_(accumulated_examples)
        if self.gradient_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_norm=self.gradient_clip_norm,
                error_if_nonfinite=True,
            )
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad(set_to_none=True)

    def _autocast_context(self):
        if self.amp:
            return torch.autocast(device_type="cuda", dtype=torch.float16)
        return nullcontext()

    def fit(
        self,
        train_loader: BatchLoader,
        validation_loader: BatchLoader,
        *,
        epochs: int,
        stop_requested: Callable[[], bool] | None = None,
        status_callback: TaskStatusCallback | None = None,
    ) -> list[dict[str, float]]:
        """Train through ``epochs`` (the total target epoch), validating each epoch."""
        if isinstance(epochs, bool) or not isinstance(epochs, int) or epochs < 1:
            raise ValueError("epochs must be a positive integer.")
        if validation_loader is None:
            raise ValueError("Trainer.fit requires a validation DataLoader.")

        self._train_loader = train_loader
        generator = getattr(train_loader, "generator", None)
        if self._pending_loader_generator_state is not None and generator is not None:
            generator.set_state(self._pending_loader_generator_state)
            self._pending_loader_generator_state = None

        history: list[dict[str, float]] = []
        self.stopped = False
        if stop_requested is not None and stop_requested():
            self.stopped = True
            if status_callback is not None:
                status_callback("Stopping", self._epoch, epochs)
                status_callback("Stopped", self._epoch, epochs)
            if self.event_callback is not None:
                self._event_emitter(
                    TrainingEvent(
                        event_type="training_stopped",
                        experiment_dir=str(self.checkpoint_dir),
                        epoch=self._epoch or None,
                        total_epochs=epochs,
                    )
                )
            return history

        for epoch in range(self.start_epoch, epochs + 1):
            if stop_requested is not None and stop_requested():
                self.stopped = True
                if status_callback is not None:
                    status_callback("Stopping", self._epoch, epochs)
                    status_callback("Stopped", self._epoch, epochs)
                if self.event_callback is not None:
                    self._event_emitter(
                        TrainingEvent(
                            event_type="training_stopped",
                            experiment_dir=str(self.checkpoint_dir),
                            epoch=self._epoch or None,
                            total_epochs=epochs,
                        )
                    )
                break
            started = perf_counter()
            if status_callback is not None:
                status_callback("Running", epoch, epochs)
            train_result = self.train_epoch(
                train_loader, epoch=epoch, total_epochs=epochs
            )
            if status_callback is not None:
                status_callback("Validating", epoch, epochs)
            validation_result = self.validate(
                validation_loader, epoch=epoch, total_epochs=epochs
            )
            learning_rate = float(self.optimizer.param_groups[0]["lr"])
            record = {
                "epoch": float(epoch),
                "train_loss": train_result["loss"],
                **{f"train_{key}": value for key, value in train_result.items() if key != "loss"},
                "val_loss": validation_result["loss"],
                **{f"val_{key}": value for key, value in validation_result.items() if key != "loss"},
                "learning_rate": learning_rate,
                "time_seconds": perf_counter() - started,
            }
            if self.monitor not in record:
                available = ", ".join(sorted(record))
                raise ValueError(
                    f"Monitored metric {self.monitor!r} is unavailable. "
                    f"Choose one of: {available}."
                )
            monitored_value = record[self.monitor]
            if not isfinite(monitored_value):
                raise FloatingPointError(f"Monitored metric {self.monitor!r} is not finite.")

            improved = self._is_improvement(monitored_value)
            if improved:
                self._best_metric = monitored_value
            should_stop = (
                self.early_stopping.step(monitored_value)
                if self.early_stopping is not None
                else False
            )
            if self.scheduler is not None:
                self.scheduler.step()

            self._epoch = epoch
            self.start_epoch = epoch + 1
            history.append(record)
            self._training_history.append(dict(record))
            self._log_epoch(record, epochs)
            if improved:
                self.save_checkpoint(
                    self.checkpoint_dir / "best_model.pth",
                    epoch=epoch,
                    metrics=record,
                )
            self.save_checkpoint(
                self.checkpoint_dir / "last_model.pth",
                epoch=epoch,
                metrics=record,
            )
            if self.event_callback is not None:
                self._event_emitter(
                    TrainingEvent(
                        event_type="epoch_completed",
                        experiment_dir=str(self.checkpoint_dir),
                        epoch=epoch,
                        total_epochs=epochs,
                        metrics=dict(record),
                    )
                )
            if should_stop:
                logger.info(
                    f"Early stopping: {self.monitor} did not improve for "
                    f"{self.early_stopping.patience} epochs."
                )
                break
            if stop_requested is not None and stop_requested() and epoch < epochs:
                self.stopped = True
                if status_callback is not None:
                    status_callback("Stopping", epoch, epochs)
                if self.event_callback is not None:
                    self._event_emitter(
                        TrainingEvent(
                            event_type="training_stopping",
                            experiment_dir=str(self.checkpoint_dir),
                            epoch=epoch,
                            total_epochs=epochs,
                            message="Stop request accepted at the completed epoch boundary.",
                        )
                    )
                    self._event_emitter(
                        TrainingEvent(
                            event_type="training_stopped",
                            experiment_dir=str(self.checkpoint_dir),
                            epoch=epoch,
                            total_epochs=epochs,
                        )
                    )
                break

        return history

    def _is_improvement(self, value: float) -> bool:
        if self._best_metric is None:
            return True
        return value < self._best_metric if self.mode == "min" else value > self._best_metric

    @staticmethod
    def _log_epoch(record: Mapping[str, float], total_epochs: int) -> None:
        epoch_fields = [f"Epoch {int(record['epoch'])}/{total_epochs}"]
        for key in ("train_loss", "val_loss"):
            epoch_fields.append(f"{key}={record[key]:.4f}")
        metric_names = sorted(
            key
            for key in record
            if key.startswith(("train_", "val_")) and key not in {"train_loss", "val_loss"}
        )
        epoch_fields.extend(f"{key}={record[key]:.4f}" for key in metric_names)
        epoch_fields.append(f"lr={record['learning_rate']:.6g}")
        epoch_fields.append(f"time={record['time_seconds']:.2f}s")
        logger.info(" | ".join(epoch_fields))

    def save_checkpoint(
        self,
        path: str | Path | None = None,
        *,
        epoch: int | None = None,
        metrics: Mapping[str, float] | None = None,
    ) -> Path:
        """Save model and training state; defaults to ``last_model.pth``."""
        checkpoint_path = Path(path) if path is not None else self.checkpoint_dir / "last_model.pth"
        try:
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as error:
            raise OSError(
                f"Could not prepare checkpoint directory {checkpoint_path.parent}: {error}"
            ) from error
        loader_generator = getattr(self._train_loader, "generator", None)
        checkpoint: dict[str, Any] = {
            "format_version": 1,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "optimizer_type": type(self.optimizer).__name__,
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
            "scheduler_type": type(self.scheduler).__name__ if self.scheduler else None,
            "amp_enabled": self.amp,
            "optimization_settings": {
                "gradient_accumulation_steps": self.gradient_accumulation_steps,
                "gradient_clip_norm": self.gradient_clip_norm,
            },
            "monitor": self.monitor,
            "mode": self.mode,
            "task_type": self.task_type,
            "label_names": list(self.label_names) if self.label_names is not None else None,
            "resume_metadata": self.resume_metadata,
            "scaler_state_dict": self.scaler.state_dict(),
            "epoch": self._epoch if epoch is None else epoch,
            "metrics": dict(metrics or {}),
            "training_history": self.training_history,
            "best_metric": self._best_metric,
            "early_stopping_state": (
                self.early_stopping.state_dict() if self.early_stopping else None
            ),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": (
                torch.cuda.get_rng_state_all() if self.device.type == "cuda" else None
            ),
            "train_loader_generator_state": (
                loader_generator.get_state() if loader_generator is not None else None
            ),
        }
        temporary_path = checkpoint_path.with_name(f"{checkpoint_path.name}.tmp")
        try:
            torch.save(checkpoint, temporary_path)
            temporary_path.replace(checkpoint_path)
        except Exception as error:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                logger.warning("Could not remove incomplete checkpoint file %s", temporary_path)
            raise OSError(f"Could not save training checkpoint to {checkpoint_path}: {error}") from error
        return checkpoint_path

    def load_checkpoint(self, path: str | Path, *, restore_rng: bool = True) -> dict[str, Any]:
        """Restore model, optimizer, scheduler, and epoch state from a checkpoint."""
        checkpoint_path = Path(path)
        try:
            checkpoint = torch.load(
                checkpoint_path,
                map_location="cpu",
                weights_only=True,
            )
        except FileNotFoundError as error:
            raise FileNotFoundError(f"Training checkpoint does not exist: {checkpoint_path}") from error
        except (EOFError, OSError, pickle.UnpicklingError, RuntimeError, ValueError) as error:
            raise ValueError(f"Could not read training checkpoint {checkpoint_path}: {error}") from error

        if not isinstance(checkpoint, Mapping):
            raise ValueError(f"Invalid training checkpoint: expected a mapping in {checkpoint_path}.")
        required = {"model_state_dict", "optimizer_state_dict", "epoch", "metrics"}
        missing = sorted(required.difference(checkpoint))
        if missing:
            raise ValueError(
                f"Invalid training checkpoint {checkpoint_path}: missing {', '.join(missing)}."
            )
        saved_scheduler = checkpoint.get("scheduler_state_dict")
        if (saved_scheduler is None) != (self.scheduler is None):
            raise ValueError("Checkpoint scheduler settings do not match this Trainer.")
        saved_scheduler_type = checkpoint.get("scheduler_type")
        current_scheduler_type = type(self.scheduler).__name__ if self.scheduler else None
        if saved_scheduler_type is not None and saved_scheduler_type != current_scheduler_type:
            raise ValueError(
                f"Checkpoint scheduler type {saved_scheduler_type!r} does not match "
                f"this Trainer's {current_scheduler_type!r}."
            )
        saved_amp = checkpoint.get("amp_enabled")
        if saved_amp is not None and saved_amp != self.amp:
            raise ValueError("Checkpoint AMP settings do not match this Trainer.")
        saved_optimization_settings = checkpoint.get("optimization_settings")
        if saved_optimization_settings is not None and saved_optimization_settings != {
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "gradient_clip_norm": self.gradient_clip_norm,
        }:
            raise ValueError("Checkpoint gradient accumulation or clipping settings do not match this Trainer.")
        saved_monitor = checkpoint.get("monitor")
        if saved_monitor is not None and saved_monitor != self.monitor:
            raise ValueError("Checkpoint monitored metric does not match this Trainer.")
        saved_mode = checkpoint.get("mode")
        if saved_mode is not None and saved_mode != self.mode:
            raise ValueError("Checkpoint monitor mode does not match this Trainer.")
        saved_task_type = checkpoint.get("task_type")
        if saved_task_type is not None and saved_task_type != self.task_type:
            raise ValueError(
                f"Checkpoint task type {saved_task_type!r} does not match "
                f"this Trainer's {self.task_type!r}."
            )
        saved_label_names = checkpoint.get("label_names")
        current_label_names = list(self.label_names) if self.label_names is not None else None
        if saved_label_names is not None and saved_label_names != current_label_names:
            raise ValueError("Checkpoint label names do not match this Trainer.")
        saved_resume_metadata = checkpoint.get("resume_metadata")
        if saved_resume_metadata is not None and not self._resume_metadata_matches(
            saved_resume_metadata, self.resume_metadata
        ):
            raise ValueError(
                "Checkpoint dataset, preprocessing, or model configuration does not "
                "match this Trainer."
            )
        if not isinstance(checkpoint["optimizer_state_dict"], Mapping):
            raise ValueError(f"Invalid optimizer state in training checkpoint {checkpoint_path}.")
        saved_optimizer_type = checkpoint.get("optimizer_type")
        if saved_optimizer_type is None and type(self.optimizer).__name__ in {"Adam", "AdamW", "SGD"}:
            saved_groups = checkpoint["optimizer_state_dict"].get("param_groups")
            if isinstance(saved_groups, list) and saved_groups and isinstance(saved_groups[0], Mapping):
                if "betas" in saved_groups[0]:
                    saved_optimizer_type = "Adam"
                elif "momentum" in saved_groups[0]:
                    saved_optimizer_type = "SGD"
        if saved_optimizer_type is not None and saved_optimizer_type != type(self.optimizer).__name__:
            raise ValueError(
                f"Checkpoint optimizer type {saved_optimizer_type!r} does not match "
                f"this Trainer's {type(self.optimizer).__name__}."
            )
        if saved_scheduler is not None and not isinstance(saved_scheduler, Mapping):
            raise ValueError(f"Invalid scheduler state in training checkpoint {checkpoint_path}.")
        if (
            isinstance(checkpoint["epoch"], bool)
            or not isinstance(checkpoint["epoch"], int)
            or checkpoint["epoch"] < 0
        ):
            raise ValueError(f"Invalid epoch value in training checkpoint {checkpoint_path}.")
        if not isinstance(checkpoint["metrics"], Mapping):
            raise ValueError(f"Invalid metrics in training checkpoint {checkpoint_path}.")
        saved_history = checkpoint.get("training_history")
        try:
            if saved_history is not None:
                if not isinstance(saved_history, list) or any(
                    not isinstance(record, Mapping)
                    or "epoch" not in record
                    or isinstance(record["epoch"], bool)
                    or not isinstance(record["epoch"], (int, float))
                    for record in saved_history
                ):
                    raise ValueError("training history must contain epoch records")
                restored_history = [
                    {str(key): float(value) for key, value in record.items()}
                    for record in saved_history
                ]
            else:
                saved_metrics = checkpoint["metrics"]
                restored_history = (
                    [{str(key): float(value) for key, value in saved_metrics.items()}]
                    if checkpoint["epoch"] > 0
                    and isinstance(saved_metrics.get("epoch"), (int, float))
                    and not isinstance(saved_metrics.get("epoch"), bool)
                    else []
                )
            if restored_history and (
                restored_history[-1]["epoch"] != float(checkpoint["epoch"])
                or any(
                    later["epoch"] <= earlier["epoch"]
                    for earlier, later in zip(restored_history, restored_history[1:])
                )
            ):
                raise ValueError("training history does not match the checkpoint epoch")
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"Invalid training history in checkpoint {checkpoint_path}: {error}"
            ) from error
        early_state = checkpoint.get("early_stopping_state")
        if (early_state is None) != (self.early_stopping is None):
            raise ValueError("Checkpoint early stopping settings do not match this Trainer.")
        if early_state is not None and not isinstance(early_state, Mapping):
            raise ValueError(f"Invalid early stopping state in training checkpoint {checkpoint_path}.")
        scaler_state = checkpoint.get("scaler_state_dict")
        if scaler_state is not None and not isinstance(scaler_state, Mapping):
            raise ValueError(f"Invalid scaler state in training checkpoint {checkpoint_path}.")
        rng_state = checkpoint.get("torch_rng_state")
        if rng_state is not None and not isinstance(rng_state, Tensor):
            raise ValueError(f"Invalid Torch RNG state in training checkpoint {checkpoint_path}.")
        cuda_rng_state = checkpoint.get("cuda_rng_state")
        if cuda_rng_state is not None and (
            not isinstance(cuda_rng_state, (list, tuple))
            or any(not isinstance(state, Tensor) for state in cuda_rng_state)
        ):
            raise ValueError(f"Invalid CUDA RNG state in training checkpoint {checkpoint_path}.")
        loader_generator_state = checkpoint.get("train_loader_generator_state")
        if loader_generator_state is not None and not isinstance(loader_generator_state, Tensor):
            raise ValueError(
                f"Invalid DataLoader RNG state in training checkpoint {checkpoint_path}."
            )

        saved_model_state = checkpoint["model_state_dict"]
        current_model_state = self.model.state_dict()
        if not isinstance(saved_model_state, Mapping):
            raise ValueError(f"Invalid model state in training checkpoint {checkpoint_path}.")
        if any(not isinstance(value, Tensor) for value in saved_model_state.values()):
            raise ValueError(f"Invalid model tensor in training checkpoint {checkpoint_path}.")
        mismatched_shapes = [
            name
            for name in current_model_state.keys() & saved_model_state.keys()
            if current_model_state[name].shape != saved_model_state[name].shape
        ]
        missing_model_keys = current_model_state.keys() - saved_model_state.keys()
        extra_model_keys = saved_model_state.keys() - current_model_state.keys()
        if mismatched_shapes or missing_model_keys or extra_model_keys:
            details = []
            if mismatched_shapes:
                details.append(f"shape mismatch for {sorted(mismatched_shapes)[0]}")
            if missing_model_keys:
                details.append(f"missing key {sorted(missing_model_keys)[0]}")
            if extra_model_keys:
                details.append(f"unexpected key {sorted(extra_model_keys)[0]}")
            raise ValueError(
                f"Checkpoint model shape mismatch in {checkpoint_path}: "
                + "; ".join(details)
                + ". The model architecture must match the saved model."
            )
        try:
            self.model.load_state_dict(saved_model_state)
        except RuntimeError as error:
            raise ValueError(
                f"Checkpoint model shape mismatch in {checkpoint_path}; "
                "the model architecture must match the saved model."
            ) from error

        try:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            if self.scheduler is not None:
                self.scheduler.load_state_dict(saved_scheduler)

            if scaler_state:
                self.scaler.load_state_dict(scaler_state)
            if self.early_stopping is not None:
                self.early_stopping.load_state_dict(early_state)
        except (AttributeError, KeyError, RuntimeError, TypeError, ValueError) as error:
            raise ValueError(
                f"Invalid training state in checkpoint {checkpoint_path}: {error}"
            ) from error

        self._epoch = int(checkpoint["epoch"])
        self.start_epoch = self._epoch + 1
        self._best_metric = checkpoint.get("best_metric")
        self._training_history = restored_history
        self._pending_loader_generator_state = loader_generator_state
        if restore_rng:
            if rng_state is not None:
                torch.set_rng_state(rng_state.cpu())
            if (
                cuda_rng_state is not None
                and self.device.type == "cuda"
                and torch.cuda.is_available()
            ):
                torch.cuda.set_rng_state_all(cuda_rng_state)
        return dict(checkpoint)

    @staticmethod
    def _resume_metadata_matches(saved: Any, current: Mapping[str, Any] | None) -> bool:
        if saved == current:
            return True
        if not isinstance(saved, Mapping) or not isinstance(current, Mapping):
            return False
        saved_dataset = saved.get("dataset")
        current_dataset = current.get("dataset")
        if (
            isinstance(saved_dataset, Mapping)
            and isinstance(current_dataset, Mapping)
            and "augmentation" not in saved_dataset
        ):
            legacy_current = dict(current)
            legacy_dataset = dict(current_dataset)
            legacy_dataset.pop("augmentation", None)
            legacy_current["dataset"] = legacy_dataset
            return saved == legacy_current
        return False

    def resume(self, checkpoint_path: str | Path | None = None) -> int:
        """Restore the latest checkpoint and return the next epoch number."""
        path = Path(checkpoint_path) if checkpoint_path is not None else self.checkpoint_dir / "last_model.pth"
        self.load_checkpoint(path)
        return self.start_epoch
