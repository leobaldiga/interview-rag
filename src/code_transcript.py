from __future__ import annotations

import json
import time
from typing import Dict, List

import requests


def code_with_llama(messages: List[Dict], llm_config: Dict) -> str:
    base_url = llm_config["base_url"].rstrip("/")
    url = f"{base_url}/chat/completions"

    payload = {
        "model": llm_config.get("model", "gemma-4-E4B-it-Q8_0.gguf"),
        "messages": messages,
        "temperature": llm_config.get("temperature", 1.0),
        "top_p": llm_config.get("top_p", 0.95),
        "max_tokens": llm_config.get("max_tokens", 8192),
        "stream": llm_config.get("stream", False),
        # Gemma 4 / llama.cpp extended sampling parameters
        "top_k": llm_config.get("top_k", 64),
        "min_p": llm_config.get("min_p", 0.0),
        "repeat_penalty": llm_config.get("repeat_penalty", 1.0),
        "seed": llm_config.get("seed", 42),
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {llm_config.get('api_key', 'no-key')}",
    }

    timeout = llm_config.get("timeout_seconds", 1800)

    try:
        t0 = time.perf_counter()
        response = requests.post(
            url,
            json=payload,
            headers=headers,
            timeout=timeout,
        )
        elapsed = time.perf_counter() - t0
    except requests.exceptions.ConnectionError as e:
        raise RuntimeError(
            f"Could not connect to llama.cpp server at {url}. "
            "Is llama-server running?"
        ) from e
    except requests.exceptions.Timeout:
        raise RuntimeError(
            f"Request to {url} timed out after {timeout}s. "
            "Consider increasing llm.timeout_seconds in settings.yaml."
        )

    if not response.ok:
        raise RuntimeError(
            f"llama.cpp server returned HTTP {response.status_code} for {url}.\n"
            f"Response body: {response.text[:500]}"
        )

    try:
        data = response.json()
    except json.JSONDecodeError as e:
        raise RuntimeError(
            f"llama.cpp server response was not valid JSON.\n"
            f"Raw response: {response.text[:500]}"
        ) from e

    try:
        content = data["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError) as e:
        raise RuntimeError(
            f"Unexpected response structure from llama.cpp server.\n"
            f"Full response: {json.dumps(data, indent=2)[:500]}"
        ) from e

    if not content:
        raise RuntimeError(
            "llama.cpp server returned an empty response. "
            "The model may have stopped early or the max_tokens limit was hit immediately."
        )

    finish_reason = data["choices"][0].get("finish_reason", "unknown")
    usage = data.get("usage", {})

    print(
        f"    ↳ {elapsed:.1f}s | "
        f"prompt={usage.get('prompt_tokens', '?')} tokens | "
        f"completion={usage.get('completion_tokens', '?')} tokens | "
        f"finish={finish_reason}"
    )

    if finish_reason == "length":
        print(
            f"    ⚠️  WARNING: output truncated (finish_reason=length). "
            f"Consider increasing max_tokens in settings.yaml "
            f"(current: {llm_config.get('max_tokens', 8192)})"
        )

    return content
