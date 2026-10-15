"""Model construction and task-appropriate loss helpers."""

from uatf.models.factory import (
    MODEL_ADAPTERS,
    MODEL_REGISTRY,
    ModelAdapter,
    build_model_from_config,
    get_model,
    get_model_adapter,
    get_model_adapters,
    register_model,
)
from uatf.models.losses import (
    LOSS_REGISTRY,
    LossAdapter,
    get_loss,
    get_loss_adapter,
    register_loss,
)

__all__ = [
    "LOSS_REGISTRY",
    "MODEL_ADAPTERS",
    "MODEL_REGISTRY",
    "LossAdapter",
    "ModelAdapter",
    "build_model_from_config",
    "get_loss",
    "get_loss_adapter",
    "get_model_adapter",
    "get_model_adapters",
    "get_model",
    "register_model",
    "register_loss",
]
