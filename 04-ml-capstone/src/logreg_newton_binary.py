"""
logreg_newton_binary.py – Binary logistic regression trained with a
Newton-like update (uses the Hessian inverse via numpy.linalg.pinv).

Refactored from course Appendix 1.

Improvements over the original:
  - No hard-coded file names or paths; data is passed in as numpy arrays.
  - Accurate module name reflecting the Newton / quasi-Newton solver.
  - Results averaged over multiple random train/test splits.
  - Returns structured metrics dict compatible with run.py.
  - Deterministic via random_state.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
from sklearn.model_selection import train_test_split

from metrics import compute_metrics, get_confusion_matrix


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


def _newton_fit(
    X: np.ndarray,
    y: np.ndarray,
    max_iter: int = 300,
    tol: float = 1e-6,
) -> np.ndarray:
    """Fit binary logistic regression weights using Newton-Raphson updates.

    Parameters
    ----------
    X : shape (n_samples, n_features)  – bias column NOT included (added here)
    y : shape (n_samples,) with values in {0, 1}

    Returns
    -------
    w : shape (n_features + 1,)  – weights including bias at index 0
    """
    n, d = X.shape
    Xb = np.hstack([np.ones((n, 1)), X])  # add bias term
    w = np.zeros(Xb.shape[1])

    for _ in range(max_iter):
        p = _sigmoid(Xb @ w)
        # Gradient: X^T (p - y)
        grad = Xb.T @ (p - y)
        # Hessian: X^T diag(p*(1-p)) X
        s = p * (1.0 - p)
        H = Xb.T @ (Xb * s[:, np.newaxis])
        # Newton step: w <- w - H^{-1} grad
        delta = np.linalg.pinv(H) @ grad
        w = w - delta
        if np.linalg.norm(delta) < tol:
            break

    return w


def _predict(X: np.ndarray, w: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    n = X.shape[0]
    Xb = np.hstack([np.ones((n, 1)), X])
    return (_sigmoid(Xb @ w) >= threshold).astype(int)


def run(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    repeats: int = 10,
    max_iter: int = 300,
    tol: float = 1e-6,
    threshold: float = 0.5,
    random_state: int = 42,
) -> Dict:
    """Run binary Newton logistic regression over multiple random splits.

    Returns the metrics averaged across repeats plus the final run's
    confusion matrix and weights.
    """
    classes = np.unique(y)
    if len(classes) != 2:
        raise ValueError(
            f"logreg_newton_binary requires exactly 2 classes; got {classes.tolist()}"
        )

    all_metrics: List[Dict] = []
    rng = np.random.RandomState(random_state)

    final_cm: np.ndarray | None = None
    final_w: np.ndarray | None = None

    for i in range(repeats):
        seed = int(rng.randint(0, 2**31))
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=seed
        )
        w = _newton_fit(X_train, y_train, max_iter=max_iter, tol=tol)
        y_pred = _predict(X_test, w, threshold=threshold)
        m = compute_metrics(y_test, y_pred)
        all_metrics.append(m)
        if i == repeats - 1:
            final_cm = get_confusion_matrix(y_test, y_pred)
            final_w = w

    # Average numeric metrics across repeats
    avg_metrics: Dict = {}
    for key in all_metrics[0]:
        avg_metrics[key] = float(np.mean([m[key] for m in all_metrics]))

    avg_metrics["repeats"] = repeats
    avg_metrics["best_params"] = {
        "max_iter": max_iter,
        "tol": tol,
        "threshold": threshold,
        "test_size": test_size,
    }

    print("[logreg_newton_binary] Mean accuracy over "
          f"{repeats} splits: {avg_metrics['accuracy']:.4f}")

    return {
        "metrics": avg_metrics,
        "confusion_matrix": final_cm,
        "weights": final_w,
        "all_split_metrics": all_metrics,
    }
