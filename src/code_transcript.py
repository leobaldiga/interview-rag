from __future__ import annotations

from typing import Dict, List
import requests


def code_with_llama(messages: List[Dict], llm_config: Dict) -> str:
    base_url = llm_config["base_url"].rstrip("/")
    url = f"{base_url}/chat/completions"

    payload = {
        "model": llm_config.get("model", "not-used"),
        "messages": messages,
        "temperature": llm_config.get("temperature", 0.35),
        "top_p": llm_config.get("top_p", 0.9),
        "max_tokens": llm_config.get("max_tokens", 4096),
        "stream": llm_config.get("stream", False),
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {llm_config.get('api_key', 'no-key')}",
    }

    response = requests.post(
        url,
        json=payload,
        headers=headers,
        timeout=llm_config.get("timeout_seconds", 1800),
    )
    response.raise_for_status()

    data = response.json()
    return data["choices"][0]["message"]["content"].strip()
