from datetime import datetime
from pathlib import Path
import os
import requests

API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
API_URL = "https://api.deepseek.com/chat/completions"

ZERO_SHOT_PROMPT = """Classify the following text as "positive" or "negative".
Reply with ONLY one word: positive or negative.

Text: {text}"""

FEW_SHOT_PROMPT = """Classify the following text as "positive" or "negative".
Reply with ONLY one word: positive or negative.

Examples:
Text: "I love this product!" -> positive
Text: "Horrible, never again." -> negative
Text: "Best thing I ever bought." -> positive
Text: "Broke after one use." -> negative

Text: {text}"""


def build_prompt(text: str, mode: str = "zero") -> str:
    if mode == "few":
        return FEW_SHOT_PROMPT.format(text=text)
    return ZERO_SHOT_PROMPT.format(text=text)


def parse_prediction(raw: str) -> str:
    raw = (raw or "").strip().lower().rstrip(".")
    if "positive" in raw:
        return "positive"
    if "negative" in raw:
        return "negative"
    return "unknown"


def call_llm(prompt: str) -> str:
    if not API_KEY:
        raise ValueError("DEEPSEEK_API_KEY environment variable is not set")

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": "deepseek-chat",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": 10,
    }

    try:
        resp = requests.post(API_URL, headers=headers, json=payload, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]
    except requests.exceptions.HTTPError:
        # log HTTP status + body (e.g., 402)
        Path("outputs").mkdir(exist_ok=True)
        Path("outputs/errors.log").write_text(
            f"[{datetime.now().isoformat()}] HTTP {resp.status_code}: {resp.text}\n",
            encoding="utf-8",
        )
        return "unknown"
    except requests.exceptions.RequestException as e:
        Path("outputs").mkdir(exist_ok=True)
        Path("outputs/errors.log").write_text(
            f"[{datetime.now().isoformat()}] RequestException: {repr(e)}\n",
            encoding="utf-8",
        )
        return "unknown"

def classify_text(text: str, mode: str = "zero") -> str:
    prompt = build_prompt(text, mode)
    raw = call_llm(prompt)
    return parse_prediction(raw)