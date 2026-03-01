"""
logreg_ovo_multiclass.py – Multiclass logistic regression using a
One-vs-One (OvO) strategy with majority-vote combination.

Refactored from course Appendix 2.

Each pair of classes trains a Newton-logistic binary classifier. At
prediction time every binary classifier casts a vote for one of its two
classes; the class with the most votes wins.

Improvements over the original:
  - No hard-coded file names; data passed as numpy arrays.
  - Exposes max_iter, tol, test_size, random_state as parameters.
  - Reports macro/micro/weighted F1 (not just class-1 metrics).
  - Returns structured result dict compatible with run.py.
"""

from __future__ import annotations

from itertools import combinations
from typing import Dict, List, Tuple

import numpy as np
from sklearn.model_selection import train_test_split
from scipy.stats import mode as scipy_mode

from metrics import compute_metrics, get_confusion_matrix
from logreg_newton_binary import _newton_fit, _sigmoid


def _ovo_fit(
    X: np.ndarray,
    y: np.ndarray,
    max_iter: int = 300,
    tol: float = 1e-6,
) -> List[Tuple[int, int, np.ndarray]]:
    """Train OvO binary classifiers for every pair of classes.

    Returns a list of (class_a, class_b, weights) tuples.
    """
    classes = np.unique(y)
    classifiers: List[Tuple[int, int, np.ndarray]] = []
    for ca, cb in combinations(classes, 2):
        mask = (y == ca) | (y == cb)
        X_sub = X[mask]
        y_sub = (y[mask] == cb).astype(int)  # 0=ca, 1=cb
        w = _newton_fit(X_sub, y_sub, max_iter=max_iter, tol=tol)
        classifiers.append((int(ca), int(cb), w))
    return classifiers


def _ovo_predict(
    X: np.ndarray,
    classifiers: List[Tuple[int, int, np.ndarray]],
) -> np.ndarray:
    """Predict labels via majority vote across all OvO binary classifiers."""
    n = X.shape[0]
    votes = np.zeros((n, len(classifiers)), dtype=int)
    for i, (ca, cb, w) in enumerate(classifiers):
        n_samp = X.shape[0]
        Xb = np.hstack([np.ones((n_samp, 1)), X])
        prob_cb = _sigmoid(Xb @ w)
        votes[:, i] = np.where(prob_cb >= 0.5, cb, ca)

    # Majority vote: pick the class that appears most often per sample
    result = np.zeros(n, dtype=int)
    for j in range(n):
        mode_result = scipy_mode(votes[j], keepdims=False)
        result[j] = int(mode_result.mode)
    return result


def run(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    max_iter: int = 300,
    tol: float = 1e-6,
    random_state: int = 42,
) -> Dict:
    """Train OvO multiclass logistic regression and evaluate on a held-out set."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    classifiers = _ovo_fit(X_train, y_train, max_iter=max_iter, tol=tol)
    y_pred = _ovo_predict(X_test, classifiers)

    m = compute_metrics(y_test, y_pred)
    cm = get_confusion_matrix(y_test, y_pred)

    m["best_params"] = {"max_iter": max_iter, "tol": tol, "test_size": test_size}
    print(f"[logreg_ovo_multiclass] Accuracy: {m['accuracy']:.4f}  "
          f"F1 macro: {m['f1_macro']:.4f}")

    return {
        "metrics": m,
        "confusion_matrix": cm,
        "classifiers": classifiers,
    }
