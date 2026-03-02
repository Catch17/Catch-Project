import pandas as pd
import pytest


@pytest.fixture()
def small_phishing_csv(tmp_path):
    # Tiny synthetic dataset (numeric only) + Result labels
    df = pd.DataFrame(
        {
            "f1": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
            "f2": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "f3": [10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
            "Result": [-1, 1, -1, 1, -1, 1, -1, 1, -1, 1],
        }
    )
    path = tmp_path / "tiny.csv"
    df.to_csv(path, index=False)
    return path