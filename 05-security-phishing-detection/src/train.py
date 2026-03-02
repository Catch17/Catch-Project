import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC


def build_model(model_name: str):
    model_name = model_name.lower().strip()
    if model_name == "lr":
        return LogisticRegression(max_iter=2000, class_weight="balanced")
    if model_name == "svm":
        return SVC(kernel="rbf", probability=True, class_weight="balanced")
    raise ValueError("model must be one of: lr, svm")


def map_result_to_binary(y_raw: pd.Series) -> pd.Series:
    """
    Map Result labels to binary:
    - If labels are in {-1, 1}: -1 -> 0, 1 -> 1
    - Otherwise: treat 0 as 0, any non-zero as 1
    """
    unique_vals = set(pd.unique(y_raw))
    if unique_vals.issubset({-1, 1}):
        return (y_raw != -1).astype(int)

    y = pd.Series(y_raw).astype(int)
    return (y != 0).astype(int)


def train_and_evaluate(
    *,
    data_path: Path,
    label_col: str,
    model: str,
    outdir: Path,
    test_size: float = 0.2,
    seed: int = 42,
) -> dict:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_path)

    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' not found. Available columns: {df.columns.tolist()}")

    y_raw = df[label_col]
    X = df.drop(columns=[label_col])

    y = map_result_to_binary(y_raw)

    # numeric/categorical split
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in numeric_cols]

    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    cat_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    pre = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_cols),
            ("cat", cat_pipe, cat_cols),
        ]
    )

    clf = build_model(model)
    pipe = Pipeline(steps=[("pre", pre), ("clf", clf)])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )

    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    # probabilities for ROC-AUC
    if hasattr(pipe, "predict_proba"):
        y_score = pipe.predict_proba(X_test)[:, 1]
    else:
        y_score = pipe.decision_function(X_test)

    metrics = {
        "model": model,
        "n_rows": int(df.shape[0]),
        "n_features": int(X.shape[1]),
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_score)),
        "classification_report": classification_report(y_test, y_pred, digits=4, zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }

    (outdir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    # Confusion matrix plot
    plt.figure()
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
    plt.tight_layout()
    plt.savefig(outdir / "confusion_matrix.png", dpi=200)
    plt.close()

    # ROC curve plot
    plt.figure()
    RocCurveDisplay.from_predictions(y_test, y_score)
    plt.tight_layout()
    plt.savefig(outdir / "roc_curve.png", dpi=200)
    plt.close()

    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to CSV file")
    parser.add_argument("--label-col", default="Result", help="Label column name (default: Result)")
    parser.add_argument("--model", default="lr", choices=["lr", "svm"], help="Model type")
    parser.add_argument("--outdir", default="outputs", help="Output directory")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    metrics = train_and_evaluate(
        data_path=Path(args.data),
        label_col=args.label_col,
        model=args.model,
        outdir=Path(args.outdir),
        test_size=args.test_size,
        seed=args.seed,
    )

    # keep original stdout behavior
    print(json.dumps({k: v for k, v in metrics.items() if k != "classification_report"}, indent=2))
    print(metrics["classification_report"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
