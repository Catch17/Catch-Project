"""
run.py – Unified CLI entrypoint for the 04-ml-capstone project.

Usage example:
    python run.py --mat-path dataset.mat --mat-key X --label-col last \
                  --model svm --tune --test-size 0.2 --random-state 42

Outputs are written to 04-ml-capstone/reports/<run-id>/:
  metrics.json   – metric values and best params for this run
  summary.csv    – one row appended per run (for comparison)
  confusion_matrix.png
  tree.png       – decision_tree model only
"""

from __future__ import annotations

import argparse
import csv
import datetime
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np

# Allow running directly as a script from the src/ directory
_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from data import load_mat  # noqa: E402
from metrics import get_confusion_matrix  # noqa: E402
from plots import save_confusion_matrix, save_tree_plot  # noqa: E402

_REPORTS_DIR = Path(_SRC_DIR).parent / "reports"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_run_dir(run_id: str | None) -> Path:
    if run_id is None:
        run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = _REPORTS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _save_metrics_json(run_dir: Path, metrics: Dict[str, Any]) -> None:
    path = run_dir / "metrics.json"
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2, default=str)
    print(f"[run] Metrics saved to '{path}'")


def _append_summary_csv(run_dir: Path, model: str, metrics: Dict[str, Any]) -> None:
    summary_path = _REPORTS_DIR / "summary.csv"
    row: Dict[str, Any] = {
        "run_id": run_dir.name,
        "model": model,
        **{k: v for k, v in metrics.items() if not isinstance(v, dict)},
    }
    write_header = not summary_path.exists()
    with open(summary_path, "a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)
    print(f"[run] Summary row appended to '{summary_path}'")


# ---------------------------------------------------------------------------
# Model dispatch
# ---------------------------------------------------------------------------

def _run_logreg_binary(args: argparse.Namespace, X: np.ndarray, y: np.ndarray) -> Dict:
    import logreg_newton_binary
    return logreg_newton_binary.run(
        X, y,
        test_size=args.test_size,
        repeats=args.repeats,
        max_iter=args.max_iter,
        random_state=args.random_state,
    )


def _run_logreg_ovo(args: argparse.Namespace, X: np.ndarray, y: np.ndarray) -> Dict:
    import logreg_ovo_multiclass
    return logreg_ovo_multiclass.run(
        X, y,
        test_size=args.test_size,
        max_iter=args.max_iter,
        random_state=args.random_state,
    )


def _run_decision_tree(args: argparse.Namespace, X: np.ndarray, y: np.ndarray) -> Dict:
    import decision_tree_tuning
    return decision_tree_tuning.run(
        X, y,
        test_size=args.test_size,
        do_tune=args.tune,
        random_state=args.random_state,
    )


def _run_svm(args: argparse.Namespace, X: np.ndarray, y: np.ndarray) -> Dict:
    import svm_tuning
    return svm_tuning.run(
        X, y,
        test_size=args.test_size,
        do_tune=args.tune,
        random_state=args.random_state,
    )


_RUNNERS = {
    "logreg_binary": _run_logreg_binary,
    "logreg_ovo": _run_logreg_ovo,
    "decision_tree": _run_decision_tree,
    "svm": _run_svm,
}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run ML capstone experiments from a .mat dataset file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--mat-path", required=True,
        help="Path to the .mat dataset file.",
    )
    parser.add_argument(
        "--mat-key", default=None,
        help="Variable key inside the .mat file. Auto-detected if omitted.",
    )
    parser.add_argument(
        "--label-col", default="last",
        help="Column to use as the label: 'first', 'last', or integer index.",
    )
    parser.add_argument(
        "--model",
        choices=list(_RUNNERS.keys()),
        default="svm",
        help="Model to run.",
    )
    parser.add_argument(
        "--tune", action="store_true",
        help="Enable hyper-parameter search via cross-validation.",
    )
    parser.add_argument(
        "--test-size", type=float, default=0.2,
        help="Fraction of data to hold out as the test set.",
    )
    parser.add_argument(
        "--random-state", type=int, default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--run-id", default=None,
        help="Sub-folder name under reports/. Auto-generated timestamp if omitted.",
    )
    parser.add_argument(
        "--repeats", type=int, default=10,
        help="(logreg_binary only) Number of random train/test splits.",
    )
    parser.add_argument(
        "--max-iter", type=int, default=300,
        help="(logreg_binary / logreg_ovo) Max Newton iterations.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    # Load data
    label_col: str | int = args.label_col
    try:
        label_col = int(label_col)
    except (TypeError, ValueError):
        pass  # keep as string 'first'/'last'

    X, y = load_mat(args.mat_path, mat_key=args.mat_key, label_col=label_col)

    # Run selected model
    runner = _RUNNERS[args.model]
    result = runner(args, X, y)

    metrics: Dict = result["metrics"]
    cm: np.ndarray = result["confusion_matrix"]

    # Persist outputs
    run_dir = _make_run_dir(args.run_id)
    _save_metrics_json(run_dir, metrics)
    _append_summary_csv(run_dir, args.model, metrics)
    save_confusion_matrix(cm, str(run_dir / "confusion_matrix.png"))

    if args.model == "decision_tree" and "model" in result:
        save_tree_plot(result["model"], str(run_dir / "tree.png"))


if __name__ == "__main__":
    main()
