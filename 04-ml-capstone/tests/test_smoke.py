"""
test_smoke.py – Smoke tests for the 04-ml-capstone package.

These tests do NOT require any .mat data files.  They generate small
synthetic datasets and verify that each model runner executes end-to-end
and returns the expected keys.
"""

from __future__ import annotations

import os
import sys
import tempfile

import numpy as np
import pytest

# Make src/ importable when running pytest from the tests/ directory or repo root
_SRC_DIR = os.path.join(os.path.dirname(__file__), "..", "src")
sys.path.insert(0, os.path.abspath(_SRC_DIR))


# ---------------------------------------------------------------------------
# Helper: tiny synthetic datasets
# ---------------------------------------------------------------------------

def _binary_dataset(n: int = 80, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.RandomState(seed)
    X = rng.randn(n, 4)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    return X, y


def _multiclass_dataset(n: int = 120, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.RandomState(seed)
    X = rng.randn(n, 4)
    y = np.array([0] * (n // 3) + [1] * (n // 3) + [2] * (n - 2 * (n // 3)))
    rng.shuffle(y)
    return X, y


# ---------------------------------------------------------------------------
# metrics.py
# ---------------------------------------------------------------------------

class TestMetrics:
    def test_binary_keys(self):
        from metrics import compute_metrics
        X, y = _binary_dataset()
        m = compute_metrics(y, y)  # perfect predictions
        assert m["accuracy"] == pytest.approx(1.0)
        assert "f1_macro" in m
        assert "precision_class1" in m

    def test_multiclass_keys(self):
        from metrics import compute_metrics
        _, y = _multiclass_dataset()
        m = compute_metrics(y, y)
        assert m["accuracy"] == pytest.approx(1.0)
        assert "f1_macro" in m
        assert "f1_weighted" in m
        assert "precision_class1" not in m  # only for binary


# ---------------------------------------------------------------------------
# data.py
# ---------------------------------------------------------------------------

class TestData:
    def test_load_mat_label_last(self):
        import scipy.io
        from data import load_mat

        data = np.column_stack([np.random.randn(50, 3), np.random.randint(0, 2, 50)])
        with tempfile.NamedTemporaryFile(suffix=".mat", delete=False) as f:
            tmp_path = f.name
        try:
            scipy.io.savemat(tmp_path, {"mydata": data})
            X, y = load_mat(tmp_path, mat_key="mydata", label_col="last")
            assert X.shape == (50, 3)
            assert y.shape == (50,)
        finally:
            os.unlink(tmp_path)

    def test_load_mat_label_first(self):
        import scipy.io
        from data import load_mat

        data = np.column_stack([np.random.randint(0, 2, 50), np.random.randn(50, 3)])
        with tempfile.NamedTemporaryFile(suffix=".mat", delete=False) as f:
            tmp_path = f.name
        try:
            scipy.io.savemat(tmp_path, {"mydata": data})
            X, y = load_mat(tmp_path, mat_key="mydata", label_col="first")
            assert X.shape == (50, 3)
            assert y.shape == (50,)
        finally:
            os.unlink(tmp_path)

    def test_load_mat_auto_key(self):
        import scipy.io
        from data import load_mat

        data = np.random.randn(30, 5)
        with tempfile.NamedTemporaryFile(suffix=".mat", delete=False) as f:
            tmp_path = f.name
        try:
            scipy.io.savemat(tmp_path, {"autokey": data})
            X, y = load_mat(tmp_path)  # no mat_key
            assert X.shape[0] == 30
        finally:
            os.unlink(tmp_path)


# ---------------------------------------------------------------------------
# logreg_newton_binary
# ---------------------------------------------------------------------------

class TestLogregNewtonBinary:
    def test_run_returns_keys(self):
        from logreg_newton_binary import run
        X, y = _binary_dataset()
        result = run(X, y, repeats=2, max_iter=50, random_state=0)
        assert "metrics" in result
        assert "confusion_matrix" in result
        assert result["metrics"]["accuracy"] >= 0.0

    def test_non_binary_raises(self):
        from logreg_newton_binary import run
        X, y = _multiclass_dataset()
        with pytest.raises(ValueError, match="2 classes"):
            run(X, y, repeats=1, max_iter=5)


# ---------------------------------------------------------------------------
# logreg_ovo_multiclass
# ---------------------------------------------------------------------------

class TestLogregOvoMulticlass:
    def test_run_returns_keys(self):
        from logreg_ovo_multiclass import run
        X, y = _multiclass_dataset()
        result = run(X, y, max_iter=50, random_state=0)
        assert "metrics" in result
        assert result["metrics"]["accuracy"] >= 0.0


# ---------------------------------------------------------------------------
# decision_tree_tuning
# ---------------------------------------------------------------------------

class TestDecisionTreeTuning:
    def test_run_no_tune(self):
        from decision_tree_tuning import run
        X, y = _multiclass_dataset()
        result = run(X, y, do_tune=False, random_state=0)
        assert "model" in result
        assert result["metrics"]["accuracy"] >= 0.0

    def test_run_with_tune(self):
        from decision_tree_tuning import run
        X, y = _multiclass_dataset()
        # Use a tiny grid to keep the test fast
        grid = {"min_samples_leaf": [1, 5], "max_leaf_nodes": [None, 10]}
        result = run(X, y, do_tune=True, param_grid=grid, random_state=0)
        assert "cv_score" in result["metrics"]


# ---------------------------------------------------------------------------
# svm_tuning
# ---------------------------------------------------------------------------

class TestSvmTuning:
    def test_run_no_tune(self):
        from svm_tuning import run
        X, y = _multiclass_dataset()
        result = run(X, y, do_tune=False, random_state=0)
        assert "model" in result
        assert result["metrics"]["accuracy"] >= 0.0

    def test_run_with_tune(self):
        from svm_tuning import run
        X, y = _binary_dataset()
        grid = {"C": [0.1, 1], "gamma": ["scale"]}
        result = run(X, y, do_tune=True, param_grid=grid, random_state=0)
        assert "cv_score" in result["metrics"]


# ---------------------------------------------------------------------------
# plots.py
# ---------------------------------------------------------------------------

class TestPlots:
    def test_save_confusion_matrix(self):
        from plots import save_confusion_matrix
        cm = np.array([[10, 2], [3, 15]])
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "cm.png")
            save_confusion_matrix(cm, out)
            assert os.path.exists(out)
