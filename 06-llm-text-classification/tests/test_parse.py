from src.classify import parse_prediction


def test_parse_positive():
    assert parse_prediction("positive") == "positive"
    assert parse_prediction("Positive.") == "positive"
    assert parse_prediction("  POSITIVE  ") == "positive"


def test_parse_negative():
    assert parse_prediction("negative") == "negative"
    assert parse_prediction("Negative.") == "negative"


def test_parse_unknown():
    assert parse_prediction("I'm not sure") == "unknown"
    assert parse_prediction("") == "unknown"


def test_parse_with_extra_text():
    assert parse_prediction("The sentiment is positive") == "positive"
    assert parse_prediction("negative sentiment") == "negative"