"""
inspect_mat.py – CLI tool to inspect a .mat file.

Usage:
    python inspect_mat.py --mat-path /path/to/file.mat
"""

from __future__ import annotations

import argparse
import sys
from typing import Any

import numpy as np
import scipy.io


_METADATA_KEYS = {"__header__", "__version__", "__globals__"}


def inspect(mat_path: str) -> None:
    """Print keys, dtypes, and shapes of variables stored in *mat_path*."""
    try:
        mat = scipy.io.loadmat(mat_path)
    except Exception as exc:
        print(f"ERROR: Could not load '{mat_path}': {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"Keys in {mat_path}")
    data_keys: list[str] = []
    for key, val in mat.items():
        if key in _METADATA_KEYS:
            print(f"  {key!r:<20s}: (metadata, skipped)")
            continue
        arr: Any = val
        if isinstance(arr, np.ndarray):
            shape_str = str(arr.shape)
            dtype_str = str(arr.dtype)
        else:
            shape_str = type(arr).__name__
            dtype_str = ""
        print(f"  {key!r:<20s}: shape={shape_str}  dtype={dtype_str}")
        data_keys.append(key)

    if not data_keys:
        print("  (no non-metadata keys found)")
        return

    # Suggest sensible defaults for the run.py CLI
    suggested_key = data_keys[0]
    print()
    print("Suggested defaults:")
    print(f"  --mat-key  {suggested_key}")
    val = mat[suggested_key]
    if isinstance(val, np.ndarray) and val.ndim == 2:
        ncols = val.shape[1]
        print(f"  Array has {ncols} columns. Label is likely the first or last column.")
        print(f"  --label-col last   (column index {ncols - 1})")
        print(f"  --label-col first  (column index 0)")
    print()
    print("Example command:")
    print(
        f"  python run.py --mat-path {mat_path!r} "
        f"--mat-key {suggested_key} --label-col last --model svm --tune"
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Inspect keys and shapes of a .mat file."
    )
    parser.add_argument("--mat-path", required=True, help="Path to the .mat file")
    args = parser.parse_args(argv)
    inspect(args.mat_path)


if __name__ == "__main__":
    main()
