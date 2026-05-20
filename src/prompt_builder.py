from __future__ import annotations

import json
from pathlib import Path
from typing import Dict


def load_text_file(path: str) -> str:
    p = Path(path)
    if not p.exists():
        return ""
    return p.read_text(encoding="utf-8").strip()


def build_coding_prompt(
    chunk: str,
    transcript_id: str,
    chunk_id: str,
    schema: Dict,
    settings: Dict
) -> list[dict]:
    system_prompt_path = settings["prompts"]["system_prompt_file"]
    coding_prompt_path = settings["prompts"]["coding_prompt_file"]

    system_prompt = load_text_file(system_prompt_path)
    coding_prompt = load_text_file(coding_prompt_path)

    if not coding_prompt:
        coding_prompt = (
            "Code the transcript excerpt using only the provided schema.\n"
            "Insert one or more applicable <P-S-T> tags immediately after the relevant text span.\n"
            "Use only tags supported by the schema.\n"
            "Output only the excerpt with inline tags inserted.\n"
            "Do not add commentary, summary, or explanation."
        )

    user_prompt = (
        f"TRANSCRIPT ID: {transcript_id}\n"
        f"CHUNK ID: {chunk_id}\n\n"
        "SCHEMA:\n"
        f"{json.dumps(schema, ensure_ascii=False, indent=2)}\n\n"
        "INSTRUCTIONS:\n"
        f"{coding_prompt}\n\n"
        "TRANSCRIPT EXCERPT TO CODE:\n"
        f"{chunk}"
    )

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
