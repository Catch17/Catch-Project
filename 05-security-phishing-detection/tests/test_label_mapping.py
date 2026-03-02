import pandas as pd

from src.train import map_result_to_binary


def test_map_minus1_plus1():
    y = pd.Series([-1, 1, -1, 1])
    mapped = map_result_to_binary(y)
    assert mapped.tolist() == [0, 1, 0, 1]


def test_map_zero_one():
    y = pd.Series([0, 1, 0, 1])
    mapped = map_result_to_binary(y)
    assert mapped.tolist() == [0, 1, 0, 1]


def test_map_minus1_zero_one():
    y = pd.Series([-1, 0, 1])
    mapped = map_result_to_binary(y)
    assert mapped.tolist() == [0, 1, 1]