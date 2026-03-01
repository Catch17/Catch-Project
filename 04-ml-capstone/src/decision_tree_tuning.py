"""
decision_tree_tuning.py – Decision tree with manual 5-fold CV grid search
over min_samples_leaf and max_leaf_nodes, then final train/test evaluation.

Refactored from course Appendix 3.

Improvements over the original:
  - No hard-coded file names; data passed as numpy arrays.
  - Tuning is opt-in via the *tune* parameter.
  - Reports macro/micro/weighted F1.
  - Returns structured result dict compatible with run.py.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.tree import DecisionTreeClassifier

from metrics import compute_metrics, get_confusion_matrix


_DEFAULT_PARAM_GRID: Dict[str, List[Any]] = {
    "min_samples_leaf": [1, 2, 5, 10, 20],
    "max_leaf_nodes": [None, 10, 20, 50, 100],
}


def _cv_score(
    X: np.ndarray,
    y: np.ndarray,
    params: Dict[str, Any],
    n_splits: int = 5,
    random_state: int = 42,
) -> float:
    """Return mean CV accuracy for a given parameter dict."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    scores: List[float] = []
    for train_idx, val_idx in skf.split(X, y):
        clf = DecisionTreeClassifier(random_state=random_state, **params)
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
    """Grid-search over *param_grid* using stratified k-fold CV.

    Returns (best_params, best_cv_score).
    """
    if param_grid is None:
        param_grid = _DEFAULT_PARAM_GRID

    best_params: Dict[str, Any] = {}
    best_score = -1.0

    for msl in param_grid["min_samples_leaf"]:
        for mln in param_grid["max_leaf_nodes"]:
            params = {"min_samples_leaf": msl, "max_leaf_nodes": mln}
            score = _cv_score(
                X_train, y_train, params,
                n_splits=n_splits, random_state=random_state
            )
            if score > best_score:
                best_score = score
                best_params = dict(params)

    print(
        f"[decision_tree] Best CV params: {best_params}  "
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
) -> Dict:
    """Train a decision tree and evaluate on a held-out test set.

    If *do_tune* is True, hyper-parameters are selected via CV on the
    training split before fitting the final model.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    if do_tune:
        best_params, best_cv_score = tune(
            X_train, y_train,
            param_grid=param_grid,
            n_splits=n_splits,
            random_state=random_state,
        )
    else:
        best_params = {}
        best_cv_score = None

    clf = DecisionTreeClassifier(random_state=random_state, **best_params)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    m = compute_metrics(y_test, y_pred)
    cm = get_confusion_matrix(y_test, y_pred)

    m["best_params"] = {**best_params, "test_size": test_size}
    if best_cv_score is not None:
        m["cv_score"] = best_cv_score

    print(
        f"[decision_tree] Test accuracy: {m['accuracy']:.4f}  "
        f"F1 macro: {m['f1_macro']:.4f}"
    )

    return {
        "metrics": m,
        "confusion_matrix": cm,
        "model": clf,
    }
