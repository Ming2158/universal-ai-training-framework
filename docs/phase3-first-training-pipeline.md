# Phase 3: First PyTorch Training Pipeline

This lesson trains a small CNN on MNIST, a public dataset of handwritten digit images. The script uses MNIST's official training partition, then holds out 10% of those images for validation. The official test partition is left unused.

Run from the repository root in PowerShell:

```powershell
.\.venv\Scripts\python.exe .\scripts\train_mnist.py
```

The first run downloads MNIST into `datasets/mnist/`. Training output and `best_model.pth` go to `outputs/phase3-mnist/`; both locations are ignored by Git. Use `--epochs`, `--batch-size`, `--learning-rate`, `--device`, `--data-dir`, and `--output-dir` to change the run. For example, `--device cpu` forces CPU or `--device cuda` requires an available CUDA GPU.

## Flow

```text
MNIST images and labels
  -> Dataset
  -> DataLoader batches
  -> CNN forward pass produces digit scores
  -> CrossEntropyLoss compares scores with labels
  -> backward computes gradients
  -> Adam updates model parameters
  -> validation measures the updated model without changing it
  -> best validation-accuracy checkpoint is saved and reloaded
```

## Modules

| Module | Problem and reason | Input and output | Relationship and core code |
| --- | --- | --- | --- |
| Dataset | Turns stored image files and labels into examples the program can read. | An MNIST image becomes a `1 x 28 x 28` float tensor and its digit label becomes an integer tensor. | `datasets.MNIST(..., transform=transforms.ToTensor())` is the source used by both loaders. |
| DataLoader | Supplies examples in manageable batches and shuffles training examples. | A batch is `images` with shape `[batch, 1, 28, 28]` and `labels` with shape `[batch]`. | `DataLoader(..., shuffle=True)` feeds training; validation uses `shuffle=False`. |
| Model and forward | Turns each image into one score for each of the ten digits. | Image batch in; score tensor with shape `[batch, 10]` out. | `SimpleCNN.forward()` is called by `model(images)`. The scores are raw logits; `CrossEntropyLoss` handles them directly, so the model does not add softmax. |
| Prediction | Converts the ten scores into the model's chosen digit for each image. | Score tensor in; predicted digit IDs with shape `[batch]` out. | `predictions.argmax(dim=1)` chooses the highest-scoring digit and is used to count correct validation predictions. |
| Loss | Measures how far the current scores are from the correct digit labels. | Logits and correct labels in; one scalar loss out. | `nn.CrossEntropyLoss()` provides the value whose gradients the training loop computes. |
| Backward and optimizer | Determines how each parameter should change and applies that change. | The loss builds gradients for model parameters; Adam consumes those gradients. | `optimizer.zero_grad()` clears the previous batch's gradients, `loss.backward()` computes this batch's gradients, then `optimizer.step()` updates parameters. |
| Validation | Measures generalization on examples that were not used to update parameters. | Validation batches in; mean validation loss and accuracy out. | `validate()` calls `model.eval()` and wraps inference in `torch.no_grad()`; it never calls backward or optimizer step. |
| Checkpoint | Keeps the best validation result so it can be used after training. | Model parameter dictionary plus epoch, class names, and validation metrics; saved as `best_model.pth`. | The script saves when validation accuracy improves, then calls `torch.load()` and `load_state_dict()` and evaluates the reloaded model. |

## Core Training Loop

`train_one_epoch()` shows the update cycle in executable order:

```python
optimizer.zero_grad()
predictions = model(images)
loss = loss_function(predictions, labels)
loss.backward()
optimizer.step()
```

The outer `for epoch` loop repeats a pass over the training loader. The inner `for images, labels` loop visits one batch at a time. A batch update does not happen until `optimizer.step()`.

## Modes and Device

- `model.train()` marks the model as training. Some layers behave differently in training mode; the optimizer updates parameters after each training batch.
- `model.eval()` switches those layers to evaluation behavior. It does not calculate validation by itself and does not disable gradient tracking.
- `torch.no_grad()` disables gradient recording for validation. That saves memory and makes clear that validation does not train the model.
- `device` is selected as CUDA when available, otherwise CPU. `model.to(device)` moves model parameters; `images.to(device)` and `labels.to(device)` move each batch. Operations require participating tensors and the model to be on compatible devices.
- CPU executes tensor operations on the general-purpose processor. A CUDA GPU executes supported tensor operations on the graphics processor. `.to(device)` moves tensors between them; it does not change the model architecture.

## Terms in the Code

- **Tensor:** PyTorch's container for numbers. Images, labels, predictions, loss values, parameters, and gradients are tensors.
- **Dataset:** An object that returns one input example and its target label when given an index.
- **DataLoader:** An iterator that gets examples from a dataset, groups them into batches, and can shuffle them.
- **Batch:** The group of examples processed by one inner-loop iteration and one optimizer update.
- **Epoch:** One complete pass through all training examples. Five epochs means the training loader is visited five times.
- **Forward:** The call `model(images)` that carries inputs through the CNN to produce scores.
- **Loss:** A numeric measure of prediction error for a batch. Lower training loss usually means the model fits the training examples better; validation loss checks performance on held-out examples.
- **Gradient:** A tensor PyTorch records for a parameter to describe how the loss would change if that parameter changed.
- **Backpropagation:** The operation started by `loss.backward()` that calculates gradients through the operations used in the forward pass.
- **Optimizer:** The object that uses gradients and a learning rate to update parameters. Here it is `torch.optim.Adam`.
- **Validation:** Measuring loss and accuracy on held-out examples without changing model parameters.
- **Checkpoint:** A saved snapshot of model parameters and enough metadata to identify the best epoch and its validation result.

This first script intentionally uses a small CNN with randomly initialized parameters. There are no pretrained weights, backbone, or replacement classifier head to explain yet. The model learns its image features from the MNIST training examples.
