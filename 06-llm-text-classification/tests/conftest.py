import pandas as pd
import pytest


@pytest.fixture()
def small_csv(tmp_path):
    df = pd.DataFrame({
        "text": [
            "I love this!",
            "Terrible product.",
            "Amazing quality!",
            "Worst ever.",
        ],
        "label": ["positive", "negative", "positive", "negative"],
    })
    path = tmp_path / "test.csv"
    df.to_csv(path, index=False)
    return path