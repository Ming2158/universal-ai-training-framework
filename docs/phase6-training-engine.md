# Phase 6: Training Engine

`uatf.training.Trainer` contains the reusable optimization loop. It receives a PyTorch model, a loss function, an optimizer, optional scheduler and metric functions, and train/validation data loaders. A batch only needs to yield `(inputs, targets)`, so the trainer does not depend on image folders, CSV files, or a particular model architecture.

## Basic use

```python
import torch
from torch import nn

from uatf.models import get_loss, build_model_from_config
from uatf.training import Trainer, build_early_stopping, build_scheduler

model = build_model_from_config(config)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
training_config = config["training"]

trainer = Trainer(
    model,
    get_loss(config["task"]["type"]),
    optimizer,
    scheduler=build_scheduler(optimizer, training_config.get("scheduler")),
    metrics={"accuracy": accuracy},
    device="auto",
    checkpoint_dir="outputs/experiment-1",
    monitor="val_loss",
    mode="min",
    early_stopping=build_early_stopping(training_config.get("early_stopping")),
    amp=training_config.get("amp", {}).get("enabled", False),
)
history = trainer.fit(train_loader, validation_loader, epochs=10)
```

Metric functions receive raw model predictions and targets, and return one scalar. For single-label classification, an accuracy function can use `predictions.argmax(dim=1)`. For multi-label tasks, supply a task-appropriate metric such as thresholded F1; the trainer does not impose a metric definition.

The example assumes `accuracy`, `train_loader`, and `validation_loader` have been prepared by the caller. This keeps data and task-specific metric decisions outside `Trainer`.

## Training loop

`train_epoch()` switches to training mode, computes the loss, runs backward propagation, and updates model parameters for each batch. `validate()` switches to evaluation mode and computes loss and metrics without gradients or optimizer updates. `fit()` calls both methods for each epoch, steps the scheduler once per epoch, records elapsed time and learning rate, writes checkpoints, and prints one concise epoch line.

The returned history is a list of dictionaries containing `train_loss`, `val_loss`, configured `train_*` and `val_*` metrics, `learning_rate`, and `time_seconds`. A DataLoader that produces no examples raises an invalid-dataset error. Incompatible model/target shapes, non-finite losses, and CUDA out-of-memory errors include batch context and corrective guidance.

## Checkpoints and resume

After every completed epoch, `last_model.pth` is replaced with the latest state. `best_model.pth` is updated only when the selected monitor improves. Both contain:

- `model_state_dict`: learned weights, needed to make predictions or continue training.
- `optimizer_state_dict`: momentum or adaptive optimizer statistics such as Adam's running averages; without these, resumed updates change.
- `scheduler_state_dict`: scheduler progress and current learning-rate schedule position.
- `epoch` and `metrics`: the completed epoch and its results, for resume positioning and experiment comparison.
- AMP scaler state, best-monitor value, early-stopping counter, and PyTorch RNG states when present.

`last_model.pth` is the resume point; `best_model.pth` is the model selected for evaluation or inference. To continue, construct the same model, optimizer, scheduler, and early-stopping settings, then call `trainer.resume()` before `fit()`:

```python
next_epoch = trainer.resume()  # Loads outputs/experiment-1/last_model.pth
history = trainer.fit(train_loader, validation_loader, epochs=10)
```

Checkpoints are written at epoch boundaries. If a process stops during epoch 7, the newest valid checkpoint is epoch 6 and resume starts epoch 7; the interrupted epoch is run again. The DataLoader's generator state is also saved when it has one, so shuffled sample order continues from the next epoch.

## Scheduler and early stopping

The only built-in scheduler is `StepLR`. It reduces the learning rate by `gamma` every `step_size` epochs. It can be disabled in YAML, and the same helper returns `None` when disabled:

```yaml
training:
  scheduler:
    enabled: false
    name: step_lr
    step_size: 5
    gamma: 0.1
```

Early stopping is optional and disabled by default. It stops when the monitored validation metric has failed to improve by `min_delta` for `patience` consecutive epochs. It can save time when validation performance has plateaued and can limit overfitting from continuing to train after generalization stops improving. Keeping it optional makes short experiments and baseline runs easier to interpret.

## AMP

Automatic Mixed Precision (AMP) is opt-in and requires a CUDA device. Training first uses ordinary FP32 by default. On supported NVIDIA GPUs, AMP runs many operations in FP16, which can use Tensor Cores and move less data through GPU memory; this can improve throughput and reduce memory use. A gradient scaler protects small FP16 gradients from underflow. AMP remains off by default because some operations or models may behave differently in reduced precision, and CPU training should stay on the stable FP32 path.

## Small offline integration test

Run the real CPU training, validation, best/last checkpoint, scheduler, best-model reload, and resume check without downloading a dataset:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\integration\test_training_engine.py
```
