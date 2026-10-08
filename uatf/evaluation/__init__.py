"""Evaluation metrics, whole-dataset evaluation, and result plots."""

from uatf.evaluation.evaluator import (
    EvaluationResult,
    evaluate_model,
    evaluate_training_run,
    save_evaluation_artifacts,
)
from uatf.evaluation.metrics import classification_metrics, multilabel_metrics
from uatf.evaluation.visualization import save_evaluation_plots

__all__ = [
    "EvaluationResult",
    "classification_metrics",
    "evaluate_model",
    "evaluate_training_run",
    "multilabel_metrics",
    "save_evaluation_artifacts",
    "save_evaluation_plots",
]
