"""Evaluation package for metrics, significance testing, and experiment tracking."""

from rag_engine.evaluation.tracker import (
    ExperimentRun,
    ExperimentTracker,
    compute_config_hash,
)

__all__ = [
    "ExperimentRun",
    "ExperimentTracker",
    "compute_config_hash",
]
