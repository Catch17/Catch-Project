from src.classify import build_prompt


def test_zero_shot_contains_text():
    prompt = build_prompt("I love this!", mode="zero")
    assert "I love this!" in prompt
    assert "positive" in prompt.lower() or "negative" in prompt.lower()


def test_few_shot_contains_examples():
    prompt = build_prompt("Great product", mode="few")
    assert "Examples:" in prompt
    assert "Great product" in prompt


def test_zero_shot_has_instruction():
    prompt = build_prompt("test", mode="zero")
    assert "Classify" in prompt
    assert "ONLY one word" in prompt