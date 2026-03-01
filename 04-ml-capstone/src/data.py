"""
data.py – configurable loader for .mat datasets.

Supports:
  - Auto-detecting the first non-metadata key when --mat-key is omitted.
  - Flexible label column selection: 'first', 'last', or an integer index.
  - Returns (X, y) as numpy arrays ready for sklearn estimators.
"""

from __future__ import annotations

import sys
from typing import Tuple

import numpy as np
import scipy.io


_METADATA_KEYS = {"__header__", "__version__", "__globals__"}


def load_mat(
    mat_path: str,
    mat_key: str | None = None,
    label_col: str | int = "last",
) -> Tuple[np.ndarray, np.ndarray]:
    """Load a .mat file and return (X, y).

    Parameters
    ----------
    mat_path:
        Path to the ``.mat`` file.
    mat_key:
        Variable name inside the ``.mat`` file.  If *None*, the first
        non-metadata key is used automatically.
    label_col:
        Which column to use as the target label.
        ``'last'``  → last column  (default)
        ``'first'`` → first column
        ``int``     → zero-based column index
    """
    try:
        mat = scipy.io.loadmat(mat_path)
    except Exception as exc:
        print(f"ERROR: Cannot load '{mat_path}': {exc}", file=sys.stderr)
        sys.exit(1)

    # Resolve key
    if mat_key is None:
        data_keys = [k for k in mat if k not in _METADATA_KEYS]
        if not data_keys:
            print("ERROR: No non-metadata keys found in the .mat file.", file=sys.stderr)
            sys.exit(1)
        mat_key = data_keys[0]
        print(f"[data] Auto-selected mat key: '{mat_key}'")

    if mat_key not in mat:
        available = [k for k in mat if k not in _METADATA_KEYS]
        print(
            f"ERROR: Key '{mat_key}' not found. Available keys: {available}",
            file=sys.stderr,
        )
        sys.exit(1)

    data: np.ndarray = np.asarray(mat[mat_key], dtype=float)

    if data.ndim != 2:
        print(
            f"ERROR: Expected a 2-D array under key '{mat_key}', "
            f"got shape {data.shape}.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Resolve label column
    n_cols = data.shape[1]
    if label_col == "last":
        col_idx = n_cols - 1
    elif label_col == "first":
        col_idx = 0
    else:
        try:
            col_idx = int(label_col)
        except (TypeError, ValueError):
            print(
                f"ERROR: --label-col must be 'first', 'last', or an integer; "
                f"got '{label_col}'.",
                file=sys.stderr,
            )
            sys.exit(1)
        if not (0 <= col_idx < n_cols):
            print(
                f"ERROR: --label-col {col_idx} is out of range for {n_cols} columns.",
                file=sys.stderr,
            )
            sys.exit(1)

    y = data[:, col_idx].astype(int)
    X = np.delete(data, col_idx, axis=1)

    print(
        f"[data] Loaded '{mat_key}' from '{mat_path}': "
        f"X shape={X.shape}, y shape={y.shape}, "
        f"classes={np.unique(y).tolist()}"
    )
    return X, y
