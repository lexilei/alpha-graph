"""LLM utility helpers."""

from __future__ import annotations

import json
import re


def extract_json(text: str) -> dict:
    """Robustly extract a JSON object from LLM output.

    Handles cases where the model wraps JSON in markdown code fences,
    adds preamble text, or returns slightly malformed output.
    """
    # Try direct parse first
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        pass

    # Try extracting from markdown code fence
    match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    # Try finding first { ... } block
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    raise ValueError(f"Could not extract JSON from response: {text[:200]}")
