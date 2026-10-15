"""Evaluation metrics, whole-dataset evaluation, and result plots."""

from uatf.evaluation.evaluator import (
    EvaluationResult,
    ModelPredictions,
    evaluate_model,
    evaluate_model_asset,
    evaluate_training_run,
    predict_model,
    save_evaluation_artifacts,
)
from uatf.evaluation.metrics import (
    CUSTOM_METRIC_REGISTRY,
    METRIC_REGISTRY,
    TRAINING_METRIC_REGISTRY,
    MetricAdapter,
    classification_metrics,
    metric_names_for_task,
    multilabel_metrics,
    register_metric,
    training_metric_names_for_task,
)
from uatf.evaluation.visualization import save_evaluation_plots

__all__ = [
    "EvaluationResult",
    "ModelPredictions",
    "CUSTOM_METRIC_REGISTRY",
    "MetricAdapter",
    "METRIC_REGISTRY",
    "TRAINING_METRIC_REGISTRY",
    "classification_metrics",
    "evaluate_model",
    "evaluate_model_asset",
    "evaluate_training_run",
    "multilabel_metrics",
    "metric_names_for_task",
    "predict_model",
    "register_metric",
    "save_evaluation_artifacts",
    "save_evaluation_plots",
    "training_metric_names_for_task",
]
