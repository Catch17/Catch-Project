"""
svm_tuning.py – SVM with StratifiedKFold CV parameter selection and a
separate held-out test evaluation (fixes evaluation leakage from Appendix 4).

Refactored from course Appendix 4.

Key fix: the original script trained on the full dataset and then evaluated on
the same data (leakage). This version:
  1. Splits data into train and test sets first.
  2. Selects C and gamma via StratifiedKFold CV on the training set only.
  3. Retrains on the full training set with the best parameters.
  4. Reports metrics on the held-out test set.

Other improvements:
  - gamma 'auto' mapped to 'scale' (sklearn >= 0.22 deprecation).
  - OneVsRestClassifier wrapper retained for multiclass datasets.
  - No hard-coded file names; data passed as numpy arrays.
  - Reports macro/micro/weighted F1.
  - Returns structured result dict compatible with run.py.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from metrics import compute_metrics, get_confusion_matrix


_DEFAULT_PARAM_GRID: Dict[str, List[Any]] = {
    "C": [0.01, 0.1, 1, 10, 100],
    "gamma": ["scale", 0.001, 0.01, 0.1, 1],
}


def _cv_score(
    X: np.ndarray,
    y: np.ndarray,
    C: float,
    gamma: Any,
    n_splits: int = 5,
    random_state: int = 42,
) -> float:
    """Return mean StratifiedKFold accuracy for (C, gamma)."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    scores: List[float] = []
    classes = np.unique(y)
    for train_idx, val_idx in skf.split(X, y):
        base = SVC(C=C, gamma=gamma, kernel="rbf", random_state=random_state)
        if len(classes) > 2:
            clf: Any = OneVsRestClassifier(base)
        else:
            clf = base
        clf.fit(X[train_idx], y[train_idx])
        scores.append(float(clf.score(X[val_idx], y[val_idx])))
    return float(np.mean(scores))


def tune(
    X_train: np.ndarray,
    y_train: np.ndarray,
    param_grid: Optional[Dict[str, List[Any]]] = None,
    n_splits: int = 5,
    random_state: int = 42,
) -> Tuple[Dict[str, Any], float]:
    """Grid-search (C, gamma) using StratifiedKFold CV.

    Returns (best_params, best_cv_score).
    """
    if param_grid is None:
        param_grid = _DEFAULT_PARAM_GRID

    best_params: Dict[str, Any] = {}
    best_score = -1.0

    for C in param_grid["C"]:
        for gamma in param_grid["gamma"]:
            # Map legacy 'auto' to 'scale'
            g = "scale" if gamma == "auto" else gamma
            score = _cv_score(
                X_train, y_train, C=C, gamma=g,
                n_splits=n_splits, random_state=random_state,
            )
            if score > best_score:
                best_score = score
                best_params = {"C": C, "gamma": g}

    print(
        f"[svm] Best CV params: {best_params}  "
        f"CV accuracy: {best_score:.4f}"
    )
    return best_params, best_score


def run(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    do_tune: bool = True,
    param_grid: Optional[Dict[str, List[Any]]] = None,
    n_splits: int = 5,
    random_state: int = 42,
    scale_features: bool = True,
) -> Dict:
    """Train an SVM and evaluate on a held-out test set.

    Parameters
    ----------
    scale_features:
        If True (default) features are standardised before fitting.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    if scale_features:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    if do_tune:
        best_params, best_cv_score = tune(
            X_train, y_train,
            param_grid=param_grid,
            n_splits=n_splits,
            random_state=random_state,
        )
    else:
        best_params = {"C": 1, "gamma": "scale"}
        best_cv_score = None

    classes = np.unique(y)
    base = SVC(
        C=best_params["C"],
        gamma=best_params["gamma"],
        kernel="rbf",
        random_state=random_state,
    )
    if len(classes) > 2:
        clf: Any = OneVsRestClassifier(base)
    else:
        clf = base

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    m = compute_metrics(y_test, y_pred)
    cm = get_confusion_matrix(y_test, y_pred)

    m["best_params"] = {**best_params, "test_size": test_size, "scale_features": scale_features}
    if best_cv_score is not None:
        m["cv_score"] = best_cv_score

    print(
        f"[svm] Test accuracy: {m['accuracy']:.4f}  "
        f"F1 macro: {m['f1_macro']:.4f}"
    )

    return {
        "metrics": m,
        "confusion_matrix": cm,
        "model": clf,
    }
