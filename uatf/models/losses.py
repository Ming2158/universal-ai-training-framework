"""Task-compatible loss registry for raw classification logits."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from torch import nn

from uatf.models.factory import normalize_task


@dataclass(frozen=True)
class LossAdapter:
    identifier: str
    display_name: str
    supported_tasks: tuple[str, ...]
    builder: Callable[[], nn.Module]
    input_contract: str
    target_contract: str


LOSS_REGISTRY: dict[str, LossAdapter] = {
    "cross_entropy": LossAdapter(
        identifier="cross_entropy",
        display_name="Cross Entropy",
        supported_tasks=("classification",),
        builder=nn.CrossEntropyLoss,
        input_contract="raw logits shaped [batch, classes]",
        target_contract="integer class IDs shaped [batch]",
    ),
    "bce_with_logits": LossAdapter(
        identifier="bce_with_logits",
        display_name="Binary Cross Entropy With Logits",
        supported_tasks=("multi_label",),
        builder=nn.BCEWithLogitsLoss,
        input_contract="raw logits shaped [batch, labels]",
        target_contract="float multi-hot targets shaped [batch, labels]",
    ),
}

_DEFAULT_LOSS_FOR_TASK = {
    "classification": "cross_entropy",
    "multi_label": "bce_with_logits",
}


def get_loss(task: str, name: str | None = None) -> nn.Module:
    """Build the task default or a registered loss compatible with ``task``."""
    normalized_task = normalize_task(task)
    loss_name = name or _DEFAULT_LOSS_FOR_TASK[normalized_task]
    if not isinstance(loss_name, str):
        raise ValueError("Loss name must be a string.")
    normalized_name = loss_name.strip().lower()
    if normalized_name == "auto":
        normalized_name = _DEFAULT_LOSS_FOR_TASK[normalized_task]
    try:
        adapter = LOSS_REGISTRY[normalized_name]
    except KeyError as error:
        available = ", ".join(sorted(LOSS_REGISTRY))
        raise ValueError(f"Unknown loss {loss_name!r}. Available losses: {available}.") from error
    if normalized_task not in adapter.supported_tasks:
        supported = ", ".join(adapter.supported_tasks)
        raise ValueError(
            f"Loss {adapter.identifier!r} does not support task {normalized_task!r}; "
            f"supported tasks: {supported}."
        )
    return adapter.builder()


def register_loss(
    name: str,
    builder: Callable[[], nn.Module],
    *,
    supported_tasks: tuple[str, ...],
    input_contract: str,
    target_contract: str,
    display_name: str | None = None,
) -> None:
    """Register a loss with explicit task, logits, and target compatibility."""
    if not isinstance(name, str) or not name.strip():
        raise ValueError("Loss name cannot be empty.")
    normalized = name.strip().lower()
    if normalized in LOSS_REGISTRY:
        raise ValueError(f"Loss is already registered: {normalized}")
    if not callable(builder):
        raise TypeError("Loss builder must be callable.")
    tasks = tuple(dict.fromkeys(normalize_task(task) for task in supported_tasks))
    if not tasks:
        raise ValueError("A loss adapter must support at least one task.")
    if not input_contract.strip() or not target_contract.strip():
        raise ValueError("Loss adapters must describe logits and target contracts.")
    LOSS_REGISTRY[normalized] = LossAdapter(
        identifier=normalized,
        display_name=display_name.strip() if display_name and display_name.strip() else normalized,
        supported_tasks=tasks,
        builder=builder,
        input_contract=input_contract,
        target_contract=target_contract,
    )


def get_loss_adapter(name: str, task: str | None = None) -> LossAdapter:
    if not isinstance(name, str):
        raise ValueError("Loss name must be a string.")
    normalized_name = name.strip().lower()
    if normalized_name == "auto":
        if task is None:
            raise ValueError("A task is required to resolve the automatic loss.")
        normalized_name = _DEFAULT_LOSS_FOR_TASK[normalize_task(task)]
    try:
        adapter = LOSS_REGISTRY[normalized_name]
    except KeyError as error:
        available = ", ".join(sorted(LOSS_REGISTRY))
        raise ValueError(f"Unknown loss {name!r}. Available losses: {available}.") from error
    if task is not None and normalize_task(task) not in adapter.supported_tasks:
        raise ValueError(
            f"Loss {adapter.identifier!r} is not compatible with task {normalize_task(task)!r}."
        )
    return adapter
