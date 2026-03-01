"""
metrics.py – utilities to compute and format classification metrics.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    """Return a dict of common classification metrics.

    For binary tasks the dict contains class-level precision/recall/F1.
    For multiclass tasks macro, micro, and weighted averages are included.
    """
    classes = np.unique(np.concatenate([y_true, y_pred]))
    is_binary = len(classes) == 2

    result: Dict[str, float] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
    }

    avg_types = ["macro", "micro", "weighted"]
    for avg in avg_types:
        result[f"f1_{avg}"] = float(
            f1_score(y_true, y_pred, average=avg, zero_division=0)
        )
        result[f"precision_{avg}"] = float(
            precision_score(y_true, y_pred, average=avg, zero_division=0)
        )
        result[f"recall_{avg}"] = float(
            recall_score(y_true, y_pred, average=avg, zero_division=0)
        )

    if is_binary:
        # Also expose per-class-1 metrics for backward-compat with original scripts
        pos_label = int(classes[1])
        result["precision_class1"] = float(
            precision_score(y_true, y_pred, pos_label=pos_label, zero_division=0)
        )
        result["recall_class1"] = float(
            recall_score(y_true, y_pred, pos_label=pos_label, zero_division=0)
        )
        result["f1_class1"] = float(
            f1_score(y_true, y_pred, pos_label=pos_label, zero_division=0)
        )

    return result


def get_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Return sklearn confusion matrix."""
    return confusion_matrix(y_true, y_pred)
