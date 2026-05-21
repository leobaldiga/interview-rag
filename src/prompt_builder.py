from __future__ import annotations

import json
from pathlib import Path
from typing import Dict


def load_text_file(path: str) -> str:
    p = Path(path)
    if not p.exists():
        return ""
    return p.read_text(encoding="utf-8").strip()


def build_schema_tag_list(schema: Dict) -> str:
    """
    Render all valid tertiary tags as a flat annotated list for prompt injection.
    Groups tags under their secondary code comment for readability.
    """
    lines: list[str] = []
    for primary, secondaries in schema.get("tag_hierarchy", {}).items():
        primary_label = schema.get("schema_key", {}).get(primary, primary)
        lines.append(f"\n## {primary_label} ({primary})")
        for secondary, content in secondaries.items():
            if isinstance(content, dict):
                comment = content.get("_comment", secondary)
                lines.append(f"\n# {comment}")
                for tag in content.get("T", []):
                    lines.append(tag)
    return "\n".join(lines).strip()


def build_coding_prompt(
    chunk: str,
    transcript_id: str,
    chunk_id: str,
    schema: Dict,
    settings: Dict,
) -> list[dict]:
    system_prompt_path = settings["prompts"]["system_prompt_file"]
    coding_prompt_path = settings["prompts"]["coding_prompt_file"]

    system_prompt = load_text_file(system_prompt_path)
    coding_prompt = load_text_file(coding_prompt_path)

    if not coding_prompt:
        coding_prompt = (
            "Code the transcript excerpt using only the provided schema.\n"
            "Insert one or more applicable tags immediately after the relevant speaker turn.\n"
            "Use ONLY tags listed verbatim in the schema below.\n"
            "Tags must follow this exact format: <P_PRIMARY:S_SECONDARY:T:TERTIARY-TAG-TEXT>\n"
            "Place tags at the END of the speaker turn, never mid-sentence.\n"
            "If no tag fits a turn, do not insert any tag.\n"
            "Output only the excerpt with inline tags inserted.\n"
            "Do not add commentary, summary, explanation, or end markers."
        )

    schema_tag_list = build_schema_tag_list(schema)

    # Sanity check — catch missing schema early
    if not schema_tag_list:
        raise ValueError(
            f"build_schema_tag_list returned empty output for chunk {chunk_id}. "
            "Check that schema['tag_hierarchy'] is populated."
        )

    user_prompt = (
        f"TRANSCRIPT ID: {transcript_id}\n"
        f"CHUNK ID: {chunk_id}\n\n"
        "VALID SCHEMA TAGS (copy tertiary text verbatim — do not paraphrase or invent):\n"
        f"{schema_tag_list}\n\n"
        "INSTRUCTIONS:\n"
        f"{coding_prompt}\n\n"
        "TRANSCRIPT EXCERPT TO CODE:\n"
        f"{chunk}"
    )

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
