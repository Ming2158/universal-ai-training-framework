from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from uatf.training import EarlyStopping, Trainer, build_early_stopping, build_scheduler


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


def make_trainer(checkpoint_dir: Path) -> Trainer:
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
    assert best_checkpoint["metrics"]["val_loss"] == min(
        row["val_loss"] for row in first_history
    )
    assert last_checkpoint["scheduler_state_dict"]["last_epoch"] == 2

    resumed = make_trainer(tmp_path)
    assert resumed.resume() == 3
    resumed_history = resumed.fit(train_loader, validation_loader, epochs=4)
    assert [int(row["epoch"]) for row in resumed_history] == [3, 4]

    completed_checkpoint = torch.load(last_path, map_location="cpu", weights_only=True)
    best_checkpoint = torch.load(best_path, map_location="cpu", weights_only=True)
    all_history = first_history + resumed_history
    assert completed_checkpoint["epoch"] == 4
    assert best_checkpoint["metrics"]["val_loss"] == min(row["val_loss"] for row in all_history)

    restored_best = make_trainer(tmp_path)
    restored_best.load_checkpoint(best_path)
    best_validation = restored_best.validate(validation_loader)
    assert best_validation["loss"] == pytest.approx(best_checkpoint["metrics"]["val_loss"])


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
