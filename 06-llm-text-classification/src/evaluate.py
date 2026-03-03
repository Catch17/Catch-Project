import argparse
import json
import time
from pathlib import Path

import pandas as pd

from src.classify import classify_text


def evaluate(data_path: Path, mode: str, outdir: Path) -> dict:
    """Run classification on all samples and compute metrics."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_path)

    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError("CSV must have 'text' and 'label' columns")

    predictions = []
    for _, row in df.iterrows():
        pred = classify_text(row["text"], mode=mode)
        predictions.append(pred)
        time.sleep(0.5)  # rate limit protection

    df["prediction"] = predictions

    # Compute metrics (only on known predictions)
    mask = df["prediction"] != "unknown"
    valid = df[mask]

    correct = (valid["label"] == valid["prediction"]).sum()
    total = len(valid)
    accuracy = correct / total if total > 0 else 0.0

    tp = ((valid["prediction"] == "positive") & (valid["label"] == "positive")).sum()
    fp = ((valid["prediction"] == "positive") & (valid["label"] == "negative")).sum()
    fn = ((valid["prediction"] == "negative") & (valid["label"] == "positive")).sum()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    metrics = {
        "mode": mode,
        "total_samples": len(df),
        "valid_predictions": int(total),
        "unknown_predictions": int(len(df) - total),
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }

    # Save
    (outdir / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    df.to_csv(outdir / "predictions.csv", index=False)

    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to CSV")
    parser.add_argument("--mode", default="zero", choices=["zero", "few"])
    parser.add_argument("--outdir", default="outputs")
    args = parser.parse_args()

    metrics = evaluate(
        data_path=Path(args.data),
        mode=args.mode,
        outdir=Path(args.outdir),
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()