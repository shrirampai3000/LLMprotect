"""Evaluation module initialization."""
from .metrics import MetricsCalculator, EvaluationResult
from .benchmark import BenchmarkSuite
from .visualizations import (
    plot_confusion_matrix,
    plot_roc_curve,
    plot_attention_heatmap,
    plot_training_history
)

__all__ = [
    "MetricsCalculator",
    "EvaluationResult",
    "BenchmarkSuite",
    "plot_confusion_matrix",
    "plot_roc_curve",
    "plot_attention_heatmap",
    "plot_training_history",
]
