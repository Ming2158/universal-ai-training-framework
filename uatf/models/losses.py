"""Loss functions for single-label and multi-label classification."""

from __future__ import annotations

from torch import nn

from uatf.models.factory import normalize_task


def get_loss(task: str) -> nn.Module:
    """Return a loss that accepts raw model logits and dataset targets."""
    normalized_task = normalize_task(task)
    if normalized_task == "classification":
        return nn.CrossEntropyLoss()
    return nn.BCEWithLogitsLoss()
