from __future__ import annotations

import json
from typing import Any

import ollama

OLLAMA_HOST = "http://127.0.0.1:11434"
THINK = False


def _parse_json_content(raw: str) -> dict[str, Any]:
    text = (raw or "").strip()
    if not text:
        return {}
    try:
        return json.loads(text)
    except json.JSONDecodeError as err:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                pass
        raise err


def chat_json(
    model: str,
    messages: list[dict[str, str]],
    schema: dict[str, Any],
    *,
    host: str = OLLAMA_HOST,
    temperature: float = 0.3,
    num_predict: int = 512,
    max_attempts: int = 3,
) -> dict[str, Any]:
    client = ollama.Client(host=host)
    predict = num_predict
    last_error: json.JSONDecodeError | None = None

    for _attempt in range(max_attempts):
        response = client.chat(
            model=model,
            messages=messages,
            think=THINK,
            format=schema,
            options={"temperature": temperature, "num_predict": predict},
        )
        try:
            return _parse_json_content(response.message.content or "")
        except json.JSONDecodeError as err:
            last_error = err
            predict = min(int(predict * 1.5), 4096)

    if last_error is not None:
        return {}
    return {}
