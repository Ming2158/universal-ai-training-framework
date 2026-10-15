from __future__ import annotations

import threading
from pathlib import Path

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from uatf.training import (
    EarlyStopping,
    Trainer,
    build_early_stopping,
    build_optimizer,
    build_scheduler,
)


def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return (logits.argmax(dim=1) == targets).float().mean()


def make_loaders() -> tuple[DataLoader, DataLoader]:
    generator = torch.Generator().manual_seed(13)
    inputs = torch.randn(96, 2, generator=generator)
    targets = (inputs[:, 0] + inputs[:, 1] > 0).long()
    train_loader = DataLoader(
        TensorDataset(inputs[:72], targets[:72]),
        batch_size=12,
        shuffle=True,
        generator=torch.Generator().manual_seed(21),
    )
    validation_loader = DataLoader(TensorDataset(inputs[72:], targets[72:]), batch_size=8)
    return train_loader, validation_loader


def make_trainer(
    checkpoint_dir: Path,
    *,
    event_callback=None,
    task_type: str | None = None,
    label_names: list[str] | None = None,
    resume_metadata: dict[str, object] | None = None,
) -> Trainer:
    model = nn.Linear(2, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.2, momentum=0.9)
    scheduler = build_scheduler(
        optimizer,
        {"enabled": True, "name": "step_lr", "step_size": 1, "gamma": 0.5},
    )
    return Trainer(
        model,
        nn.CrossEntropyLoss(),
        optimizer,
        scheduler=scheduler,
        metrics={"accuracy": accuracy},
        device="cpu",
        checkpoint_dir=checkpoint_dir,
        monitor="val_loss",
        mode="min",
        task_type=task_type,
        label_names=label_names,
        resume_metadata=resume_metadata,
        event_callback=event_callback,
    )


def test_training_validation_checkpoints_and_resume(tmp_path: Path) -> None:
    torch.manual_seed(7)
    train_loader, validation_loader = make_loaders()
    trainer = make_trainer(tmp_path)

    first_history = trainer.fit(train_loader, validation_loader, epochs=2)

    best_path = tmp_path / "best_model.pth"
    last_path = tmp_path / "last_model.pth"
    assert best_path.is_file()
    assert last_path.is_file()
    assert [int(row["epoch"]) for row in first_history] == [1, 2]
    assert all("train_loss" in row and "val_accuracy" in row for row in first_history)

    best_checkpoint = torch.load(best_path, map_location="cpu", weights_only=True)
    last_checkpoint = torch.load(last_path, map_location="cpu", weights_only=True)
    assert {
        "model_state_dict",
        "optimizer_state_dict",
        "scheduler_state_dict",
        "epoch",
        "metrics",
    }.issubset(best_checkpoint)
    assert last_checkpoint["epoch"] == 2
    assert last_checkpoint["optimizer_type"] == "SGD"
    assert last_checkpoint["scheduler_type"] == "StepLR"
    assert best_checkpoint["metrics"]["val_loss"] == min(
        row["val_loss"] for row in first_history
    )
    assert last_checkpoint["scheduler_state_dict"]["last_epoch"] == 2
    assert [int(row["epoch"]) for row in last_checkpoint["training_history"]] == [1, 2]

    legacy_checkpoint = dict(last_checkpoint)
    for key in ("optimizer_type", "scheduler_type", "amp_enabled", "monitor", "mode"):
        legacy_checkpoint.pop(key)
    legacy_path = tmp_path / "legacy-checkpoint.pth"
    torch.save(legacy_checkpoint, legacy_path)
    legacy_resumed = make_trainer(tmp_path)
    assert legacy_resumed.resume(legacy_path) == 3

    resumed = make_trainer(tmp_path)
    assert resumed.resume() == 3
    resumed_history = resumed.fit(train_loader, validation_loader, epochs=4)
    assert [int(row["epoch"]) for row in resumed_history] == [3, 4]
    assert [int(row["epoch"]) for row in resumed.training_history] == [1, 2, 3, 4]

    completed_checkpoint = torch.load(last_path, map_location="cpu", weights_only=True)
    best_checkpoint = torch.load(best_path, map_location="cpu", weights_only=True)
    all_history = first_history + resumed_history
    assert completed_checkpoint["epoch"] == 4
    assert best_checkpoint["metrics"]["val_loss"] == min(row["val_loss"] for row in all_history)

    restored_best = make_trainer(tmp_path)
    restored_best.load_checkpoint(best_path)
    best_validation = restored_best.validate(validation_loader)
    assert best_validation["loss"] == pytest.approx(best_checkpoint["metrics"]["val_loss"])

    mismatched_model = nn.Linear(2, 2)
    mismatched_optimizer = torch.optim.Adam(mismatched_model.parameters(), lr=0.2)
    mismatched_trainer = Trainer(
        mismatched_model,
        nn.CrossEntropyLoss(),
        mismatched_optimizer,
        scheduler=build_scheduler(
            mismatched_optimizer,
            {"enabled": True, "name": "step_lr", "step_size": 1, "gamma": 0.5},
        ),
        device="cpu",
        checkpoint_dir=tmp_path,
    )
    with pytest.raises(ValueError, match="optimizer type 'SGD' does not match.*Adam"):
        mismatched_trainer.load_checkpoint(last_path)


def test_stop_is_handled_at_epoch_boundary_and_checkpoint_resumes_complete_history(
    tmp_path: Path,
) -> None:
    train_loader, validation_loader = make_loaders()
    stop_event = threading.Event()
    events = []
    statuses: list[str] = []

    def observe(event: object) -> None:
        events.append(event)
        if getattr(event, "event_type") == "epoch_completed":
            stop_event.set()

    trainer = make_trainer(
        tmp_path,
        event_callback=observe,
        task_type="classification",
        label_names=["negative", "positive"],
        resume_metadata={"image_size": 224},
    )
    history = trainer.fit(
        train_loader,
        validation_loader,
        epochs=5,
        stop_requested=stop_event.is_set,
        status_callback=lambda status, _epoch, _total: statuses.append(status),
    )

    assert [int(row["epoch"]) for row in history] == [1]
    assert trainer.stopped is True
    assert statuses == ["Running", "Validating", "Stopping"]
    assert [event.event_type for event in events][-2:] == [
        "training_stopping",
        "training_stopped",
    ]
    checkpoint = torch.load(
        tmp_path / "last_model.pth", map_location="cpu", weights_only=True
    )
    assert checkpoint["epoch"] == 1
    assert [int(row["epoch"]) for row in checkpoint["training_history"]] == [1]

    resumed = make_trainer(
        tmp_path,
        task_type="classification",
        label_names=["negative", "positive"],
        resume_metadata={"image_size": 224},
    )
    assert resumed.resume() == 2
    assert [int(row["epoch"]) for row in resumed.training_history] == [1]
    continuation = resumed.fit(train_loader, validation_loader, epochs=3)
    assert [int(row["epoch"]) for row in continuation] == [2, 3]
    assert [int(row["epoch"]) for row in resumed.training_history] == [1, 2, 3]
    assert resumed.optimizer.param_groups[0]["lr"] == pytest.approx(0.025)

    incompatible = make_trainer(tmp_path, task_type="multi_label")
    with pytest.raises(ValueError, match="task type 'classification'.*'multi_label'"):
        incompatible.resume()
    incompatible_labels = make_trainer(
        tmp_path,
        task_type="classification",
        label_names=["positive", "negative"],
    )
    with pytest.raises(ValueError, match="label names do not match"):
        incompatible_labels.resume()
    incompatible_dataset = make_trainer(
        tmp_path,
        task_type="classification",
        label_names=["negative", "positive"],
        resume_metadata={"image_size": 512},
    )
    with pytest.raises(ValueError, match="dataset, preprocessing, or model"):
        incompatible_dataset.resume()


def test_training_progress_events_include_batch_phase_and_observer_failures_are_isolated(
    tmp_path: Path,
) -> None:
    train_loader, validation_loader = make_loaders()
    events = []
    trainer = make_trainer(tmp_path / "events", event_callback=events.append)

    history = trainer.fit(train_loader, validation_loader, epochs=1)

    progress_events = [event for event in events if event.event_type == "training_progress"]
    assert history
    assert {event.stage for event in progress_events} == {"training", "validation"}
    assert all(event.timestamp_utc for event in progress_events)
    assert all(event.batch_index is not None for event in progress_events)
    assert all(event.total_batches is not None for event in progress_events)
    assert any(
        event.stage == "training"
        and event.batch_index == event.total_batches
        and "train_loss" in event.metrics
        for event in progress_events
    )
    assert any(
        event.stage == "validation"
        and event.batch_index == event.total_batches
        and "val_loss" in event.metrics
        for event in progress_events
    )

    def failing_observer(event: object) -> None:
        raise RuntimeError("observer unavailable")

    isolated = make_trainer(
        tmp_path / "observer-failure", event_callback=failing_observer
    )
    assert isolated.fit(train_loader, validation_loader, epochs=1)


def test_early_stopping_and_optional_amp_errors(tmp_path: Path) -> None:
    stopper = EarlyStopping(patience=2, min_delta=0.01)
    assert stopper.step(1.0) is False
    assert stopper.step(0.995) is False
    assert stopper.step(1.01) is True

    model = nn.Linear(2, 2)
    with pytest.raises(ValueError, match="AMP requires a CUDA device"):
        Trainer(
            model,
            nn.CrossEntropyLoss(),
            torch.optim.SGD(model.parameters(), lr=0.1),
            device="cpu",
            checkpoint_dir=tmp_path,
            amp=True,
        )

    train_loader, validation_loader = make_loaders()
    model = nn.Linear(2, 2)
    stopped = Trainer(
        model,
        nn.CrossEntropyLoss(),
        torch.optim.SGD(model.parameters(), lr=0.0),
        device="cpu",
        checkpoint_dir=tmp_path / "early-stop",
        early_stopping=build_early_stopping({"enabled": True, "patience": 1}),
    )
    history = stopped.fit(train_loader, validation_loader, epochs=5)
    assert [int(row["epoch"]) for row in history] == [1, 2]


def test_adamw_cosine_scheduler_and_gradient_accumulation(tmp_path: Path) -> None:
    model = nn.Linear(1, 1, bias=False)
    optimizer = build_optimizer(
        model.parameters(),
        {"name": "adamw", "weight_decay": 0.02},
        learning_rate=0.01,
    )
    scheduler = build_scheduler(
        optimizer, {"enabled": True, "name": "cosine", "t_max": 4}, epochs=4
    )
    assert isinstance(optimizer, torch.optim.AdamW)
    assert isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingLR)
    assert scheduler.T_max == 4

    inputs = torch.arange(1, 6, dtype=torch.float32).reshape(-1, 1)
    targets = inputs * 3
    full_loader = DataLoader(TensorDataset(inputs, targets), batch_size=5)
    micro_loader = DataLoader(TensorDataset(inputs, targets), batch_size=2)
    full_model = nn.Linear(1, 1, bias=False)
    accumulated_model = nn.Linear(1, 1, bias=False)
    full_model.weight.data.fill_(0.25)
    accumulated_model.load_state_dict(full_model.state_dict())
    full_optimizer = torch.optim.SGD(full_model.parameters(), lr=0.001)
    accumulated_optimizer = torch.optim.SGD(accumulated_model.parameters(), lr=0.001)
    full_trainer = Trainer(
        full_model,
        nn.MSELoss(),
        full_optimizer,
        device="cpu",
        checkpoint_dir=tmp_path / "full",
    )
    accumulated_trainer = Trainer(
        accumulated_model,
        nn.MSELoss(),
        accumulated_optimizer,
        device="cpu",
        checkpoint_dir=tmp_path / "accumulated",
        gradient_accumulation_steps=4,
    )

    full_trainer.train_epoch(full_loader)
    accumulated_trainer.train_epoch(micro_loader)

    assert torch.allclose(full_model.weight, accumulated_model.weight, atol=1e-7)


def test_gradient_clipping_is_applied_before_optimizer_step(tmp_path: Path) -> None:
    inputs = torch.full((2, 1), 100.0)
    targets = torch.full((2, 1), -100.0)
    model = nn.Linear(1, 1, bias=False)
    model.weight.data.zero_()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    trainer = Trainer(
        model,
        nn.MSELoss(),
        optimizer,
        device="cpu",
        checkpoint_dir=tmp_path,
        gradient_clip_norm=0.01,
    )

    trainer.train_epoch(DataLoader(TensorDataset(inputs, targets), batch_size=2))

    assert model.weight.item() == pytest.approx(-0.001, abs=1e-6)


def test_adamw_and_cosine_states_round_trip_through_checkpoints(tmp_path: Path) -> None:
    train_loader, validation_loader = make_loaders()

    def make_adamw_trainer(checkpoint_dir: Path) -> Trainer:
        model = nn.Linear(2, 2)
        optimizer = build_optimizer(
            model.parameters(),
            {"name": "adamw", "weight_decay": 0.01},
            learning_rate=0.02,
        )
        scheduler = build_scheduler(
            optimizer,
            {"enabled": True, "name": "cosine", "t_max": 2, "eta_min": 0.0},
            epochs=2,
        )
        return Trainer(
            model,
            nn.CrossEntropyLoss(),
            optimizer,
            scheduler=scheduler,
            metrics={"accuracy": accuracy},
            device="cpu",
            checkpoint_dir=checkpoint_dir,
        )

    trainer = make_adamw_trainer(tmp_path)
    trainer.fit(train_loader, validation_loader, epochs=2)
    checkpoint = torch.load(
        tmp_path / "last_model.pth", map_location="cpu", weights_only=True
    )
    assert checkpoint["optimizer_type"] == "AdamW"
    assert checkpoint["scheduler_type"] == "CosineAnnealingLR"

    resumed = make_adamw_trainer(tmp_path)
    assert resumed.resume(tmp_path / "last_model.pth") == 3
    assert resumed.scheduler.state_dict()["last_epoch"] == 2


def test_legacy_resume_metadata_remains_compatible_with_default_augmentation() -> None:
    previous = {
        "dataset": {"image_size": 224, "mean": [0.4, 0.4, 0.4]},
        "model": {"name": "resnet50"},
    }
    current = {
        "dataset": {
            "image_size": 224,
            "mean": [0.4, 0.4, 0.4],
            "augmentation": {"enabled": False},
        },
        "model": {"name": "resnet50"},
    }

    assert Trainer._resume_metadata_matches(previous, current)
    assert not Trainer._resume_metadata_matches(
        previous | {"model": {"name": "other"}}, current
    )


def test_invalid_data_shapes_and_non_finite_loss_are_clear(tmp_path: Path) -> None:
    inputs = torch.randn(4, 2)
    targets = torch.zeros(4, dtype=torch.long)
    loader = DataLoader(TensorDataset(inputs, targets), batch_size=2)

    wrong_model = nn.Linear(3, 2)
    wrong_shape_trainer = Trainer(
        wrong_model,
        nn.CrossEntropyLoss(),
        torch.optim.SGD(wrong_model.parameters(), lr=0.1),
        device="cpu",
        checkpoint_dir=tmp_path,
    )
    with pytest.raises(ValueError, match="Model shape mismatch"):
        wrong_shape_trainer.train_epoch(loader)

    empty_loader = DataLoader(
        TensorDataset(torch.empty(0, 2), torch.empty(0, dtype=torch.long)),
        batch_size=2,
    )
    model = nn.Linear(2, 2)
    trainer = Trainer(
        model,
        nn.CrossEntropyLoss(),
        torch.optim.SGD(model.parameters(), lr=0.1),
        device="cpu",
        checkpoint_dir=tmp_path,
    )
    with pytest.raises(ValueError, match="Invalid dataset.*no examples"):
        trainer.train_epoch(empty_loader)

    class NonFiniteLoss(nn.Module):
        def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
            return predictions.sum() * float("nan")

    non_finite_model = nn.Linear(2, 2)
    non_finite_trainer = Trainer(
        non_finite_model,
        NonFiniteLoss(),
        torch.optim.SGD(non_finite_model.parameters(), lr=0.1),
        device="cpu",
        checkpoint_dir=tmp_path,
    )
    with pytest.raises(FloatingPointError, match="NaN or infinite loss"):
        non_finite_trainer.train_epoch(loader)


def test_checkpoint_errors_include_paths_and_state_context(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = nn.Linear(2, 2)
    trainer = Trainer(
        model,
        nn.CrossEntropyLoss(),
        torch.optim.SGD(model.parameters(), lr=0.1),
        device="cpu",
        checkpoint_dir=tmp_path,
    )
    checkpoint_path = tmp_path / "checkpoints" / "broken.pth"

    def fail_save(checkpoint: object, path: Path) -> None:
        raise RuntimeError("simulated disk failure")

    monkeypatch.setattr(torch, "save", fail_save)
    with pytest.raises(OSError, match="Could not save training checkpoint.*simulated disk failure"):
        trainer.save_checkpoint(checkpoint_path)
    assert not checkpoint_path.with_name("broken.pth.tmp").exists()

    malformed_checkpoint = tmp_path / "malformed.pth"
    monkeypatch.undo()
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": [],
            "epoch": 1,
            "metrics": {},
            "scheduler_state_dict": None,
            "early_stopping_state": None,
        },
        malformed_checkpoint,
    )
    with pytest.raises(ValueError, match="Invalid optimizer state.*malformed.pth"):
        trainer.load_checkpoint(malformed_checkpoint)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_z_cuda_amp_training_after_fp32_integration(tmp_path: Path) -> None:
    train_loader, validation_loader = make_loaders()
    model = nn.Linear(2, 2)
    trainer = Trainer(
        model,
        nn.CrossEntropyLoss(),
        torch.optim.Adam(model.parameters(), lr=0.01),
        metrics={"accuracy": accuracy},
        device="cuda",
        checkpoint_dir=tmp_path,
        amp=True,
    )

    training = trainer.train_epoch(train_loader)
    validation = trainer.validate(validation_loader)

    assert torch.isfinite(torch.tensor(training["loss"]))
    assert torch.isfinite(torch.tensor(validation["loss"]))
    assert next(trainer.model.parameters()).device.type == "cuda"
