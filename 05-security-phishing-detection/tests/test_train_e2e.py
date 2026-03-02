import json

from src.train import train_and_evaluate


def test_train_creates_artifacts(tmp_path, small_phishing_csv):
    outdir = tmp_path / "out"
    metrics = train_and_evaluate(
        data_path=small_phishing_csv,
        label_col="Result",
        model="lr",
        outdir=outdir,
        test_size=0.2,
        seed=42,
    )

    assert "accuracy" in metrics
    assert 0.0 <= metrics["accuracy"] <= 1.0

    metrics_path = outdir / "metrics.json"
    assert metrics_path.exists()
    assert metrics_path.stat().st_size > 0

    assert (outdir / "confusion_matrix.png").exists()
    assert (outdir / "roc_curve.png").exists()

    # JSON format sanity
    m = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert "roc_auc" in m


def test_missing_label_col_raises(tmp_path, small_phishing_csv):
    outdir = tmp_path / "out"
    try:
        train_and_evaluate(
            data_path=small_phishing_csv,
            label_col="NOT_EXIST",
            model="lr",
            outdir=outdir,
            test_size=0.2,
            seed=42,
        )
        assert False, "Expected ValueError for missing label col"
    except ValueError as e:
        assert "Label column" in str(e)


def test_reproducible_metrics(tmp_path, small_phishing_csv):
    out1 = tmp_path / "out1"
    out2 = tmp_path / "out2"

    train_and_evaluate(
        data_path=small_phishing_csv,
        label_col="Result",
        model="lr",
        outdir=out1,
        test_size=0.2,
        seed=42,
    )
    train_and_evaluate(
        data_path=small_phishing_csv,
        label_col="Result",
        model="lr",
        outdir=out2,
        test_size=0.2,
        seed=42,
    )

    m1 = json.loads((out1 / "metrics.json").read_text(encoding="utf-8"))
    m2 = json.loads((out2 / "metrics.json").read_text(encoding="utf-8"))

    assert m1["accuracy"] == m2["accuracy"]
    assert m1["roc_auc"] == m2["roc_auc"]