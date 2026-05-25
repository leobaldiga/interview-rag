from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import requests


# ---------------------------------------------------------------------------
# Tag parsing
# ---------------------------------------------------------------------------

TAG_RE = re.compile(
    r"<(P_[A-Z]+):(S_[A-Z]+):T:([^>]+)>"
)


def parse_tagged_chunks(jsonl_path: Path) -> Tuple[List[dict], List[dict]]:
    """
    Read a chunk_codes JSONL and return:
      - chunks: list of raw record dicts (with source_text, coded_text, chunk_id)
      - hits:   list of {tag, primary, secondary, quote, chunk_id}

    Quote = the speaker turn containing the tag, stripped of all tag markup.
    """
    chunks = []
    hits = []

    with jsonl_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            chunks.append(record)

            coded = record["coded_text"]
            chunk_id = record["chunk_id"]

            # Split coded text into speaker turns on INTERVIEWER/RESPONDENT labels
            turns = re.split(r"(?=(?:INTERVIEWER|RESPONDENT)\s*:)", coded)

            for turn in turns:
                tags_in_turn = TAG_RE.findall(turn)
                if not tags_in_turn:
                    continue
                # Strip all tags to get clean verbatim quote
                clean_turn = TAG_RE.sub("", turn).strip()
                # Drop the speaker label prefix for the quote
                clean_turn = re.sub(
                    r"^(?:INTERVIEWER|RESPONDENT)\s*:\s*", "", clean_turn
                ).strip()
                if not clean_turn:
                    continue
                for primary, secondary, tag_text in tags_in_turn:
                    hits.append(
                        {
                            "primary": primary,
                            "secondary": secondary,
                            "tag": tag_text.strip(),
                            "quote": clean_turn,
                            "chunk_id": chunk_id,
                        }
                    )

    return chunks, hits


# ---------------------------------------------------------------------------
# LLM calls
# ---------------------------------------------------------------------------


def _post(url: str, payload: dict, api_key: str, timeout: int) -> str:
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    resp = requests.post(url, json=payload, headers=headers, timeout=timeout)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()


def _llm_call(messages: List[dict], llm_config: dict) -> str:
    base_url = llm_config["base_url"].rstrip("/")
    url = f"{base_url}/chat/completions"
    payload = {
        "model": llm_config.get("model", "gemma-4-E4B-it-Q8_0.gguf"),
        "messages": messages,
        "temperature": llm_config.get("temperature", 1.0),
        "top_p": llm_config.get("top_p", 0.95),
        "top_k": llm_config.get("top_k", 64),
        "min_p": llm_config.get("min_p", 0.0),
        "repeat_penalty": llm_config.get("repeat_penalty", 1.0),
        "seed": llm_config.get("seed", 42),
        "max_tokens": 2048,
        "stream": False,
    }
    return _post(
        url, payload, llm_config.get("api_key", "no-key"), llm_config.get("timeout_seconds", 1800)
    )


# ---------------------------------------------------------------------------
# Header extraction
# ---------------------------------------------------------------------------

HEADER_SYSTEM = (
    "You are a qualitative research assistant. Extract factual farm profile "
    "information from interview transcript text. Be concise and factual. "
    "If information is not mentioned, write \"Not mentioned\"."
)

HEADER_USER_TMPL = """\
Read the following interview transcript and extract the following fields. \
Return ONLY a JSON object with these exact keys — no commentary, no markdown fences:

{{
  "farmer_name_or_id": "...",
  "location": "...",
  "farm_size_acres": "...",
  "land_tenure": "...",
  "years_farming": "...",
  "primary_crops": "...",
  "crop_rotation": "...",
  "cover_cropping": "...",
  "tillage_system": "...",
  "irrigation": "...",
  "other_activities": "...",
  "notable_context": "..."
}}

TRANSCRIPT:
{transcript}
"""


def extract_header(full_text: str, llm_config: dict) -> dict:
    """Call LLM to extract farm profile fields from the raw transcript."""
    # Use first ~6000 chars — enough context without blowing the context window
    excerpt = full_text[:6000]
    user_msg = HEADER_USER_TMPL.format(transcript=excerpt)
    messages = [
        {"role": "system", "content": HEADER_SYSTEM},
        {"role": "user", "content": user_msg},
    ]
    raw = _llm_call(messages, llm_config)
    # Strip accidental markdown fences if model adds them
    raw = re.sub(r"^```(?:json)?\s*", "", raw.strip())
    raw = re.sub(r"\s*```$", "", raw.strip())
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Graceful fallback — return raw string under a single key
        return {"raw_extraction": raw}


# ---------------------------------------------------------------------------
# Best-quote selection
# ---------------------------------------------------------------------------

QUOTE_SYSTEM = (
    "You are a qualitative research assistant helping summarize farmer interview data. "
    "Select the single most substantive, specific, and revealing quote for the given "
    "thematic code. Prefer quotes with concrete detail, farmer voice, or explanatory "
    "content over vague or very short statements."
)

QUOTE_USER_TMPL = """\
Thematic code: {tag}

Below are candidate quotes from the interview, each from a different passage. \
Return ONLY the text of the single best quote — verbatim, no edits, no commentary.

CANDIDATES:
{candidates}
"""


def select_best_quote(tag: str, quotes: List[str], llm_config: dict) -> str:
    """Ask the LLM to pick the best quote from candidates. Falls back to longest if only one."""
    if len(quotes) == 1:
        return quotes[0]
    numbered = "\n\n".join(f"[{i+1}] {q}" for i, q in enumerate(quotes))
    messages = [
        {"role": "system", "content": QUOTE_SYSTEM},
        {"role": "user", "content": QUOTE_USER_TMPL.format(tag=tag, candidates=numbered)},
    ]
    result = _llm_call(messages, llm_config)
    # If the model returns a bracket reference like "[2]", resolve it
    ref_match = re.match(r"^\[(\d+)\]", result.strip())
    if ref_match:
        idx = int(ref_match.group(1)) - 1
        if 0 <= idx < len(quotes):
            return quotes[idx]
    return result.strip()


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------

FIELD_LABELS = {
    "farmer_name_or_id": "Farmer ID / Name",
    "location": "Location",
    "farm_size_acres": "Farm Size (acres)",
    "land_tenure": "Land Tenure",
    "years_farming": "Years Farming",
    "primary_crops": "Primary Crops",
    "crop_rotation": "Crop Rotation",
    "cover_cropping": "Cover Cropping",
    "tillage_system": "Tillage System",
    "irrigation": "Irrigation",
    "other_activities": "Other Activities",
    "notable_context": "Notable Context",
}


def render_header_block(profile: dict, transcript_id: str) -> str:
    """Render the farm profile header section."""
    lines = [
        f"# Interview Summary: {transcript_id}",
        "",
        "## Farm Profile",
        "",
        "| Field | Value |",
        "|-------|-------|",
    ]
    if "raw_extraction" in profile:
        lines.append(f"| Raw extraction | {profile['raw_extraction']} |")
    else:
        for key, label in FIELD_LABELS.items():
            val = profile.get(key, "Not mentioned").replace("|", "\\|").replace("\n", " ")
            lines.append(f"| {label} | {val} |")
    lines.append("")
    return "\n".join(lines)


def render_code_tables(
    hits: List[dict],
    best_quotes: Dict[str, str],
    schema: dict,
) -> str:
    """Render one section per primary theme, one table per subtheme."""
    schema_key = schema.get("schema_key", {})
    hierarchy = schema.get("tag_hierarchy", {})

    # Group hits: primary -> secondary -> tag (preserve insertion order)
    grouped: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for h in hits:
        grouped[h["primary"]][h["secondary"]][h["tag"]].append(h["quote"])

    lines = ["---", "", "## Coded Theme Summary", ""]

    for primary, secondaries in hierarchy.items():
        if primary not in grouped:
            continue
        primary_label = schema_key.get(primary, primary)
        lines.append(f"### {primary_label} `{primary}`")
        lines.append("")

        for secondary, content in secondaries.items():
            if secondary not in grouped[primary]:
                continue
            comment = (
                content.get("_comment", secondary) if isinstance(content, dict) else secondary
            )
            # Use only the short label before the em-dash
            short_label = comment.split("\u2014")[0].strip() if "\u2014" in comment else comment
            lines.append(f"#### {short_label} `{secondary}`")
            lines.append("")
            lines.append("| Tag | Best Quote |")
            lines.append("|-----|-----------|")

            for tag, quotes in grouped[primary][secondary].items():
                best = best_quotes.get(tag, quotes[0])
                safe_quote = best.replace("|", "\\|").replace("\n", " ").strip()
                safe_tag = tag.replace("|", "\\|")
                lines.append(f"| {safe_tag} | \"{safe_quote}\" |")

            lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------


def generate_summary(
    jsonl_path: Path,
    raw_transcript_path: Path,
    output_path: Path,
    schema: dict,
    llm_config: dict,
    transcript_id: str,
) -> None:
    print(f"  Parsing coded chunks from {jsonl_path.name} ...")
    chunks, hits = parse_tagged_chunks(jsonl_path)

    if not hits:
        print(f"  \u26a0\ufe0f  No tags found in {jsonl_path.name} \u2014 skipping summary.")
        return

    print(f"  Found {len(hits)} tagged spans across {len(chunks)} chunks.")

    # Step 1: extract farm profile header
    print("  Extracting farm profile header ...")
    raw_text = (
        raw_transcript_path.read_text(encoding="utf-8") if raw_transcript_path.exists() else ""
    )
    profile = extract_header(raw_text, llm_config) if raw_text else {}

    # Step 2: select best quote per unique tag
    tag_quotes: dict[str, list[str]] = defaultdict(list)
    for h in hits:
        tag_quotes[h["tag"]].append(h["quote"])

    unique_tags = list(tag_quotes.keys())
    print(f"  Selecting best quotes for {len(unique_tags)} unique tags ...")
    best_quotes: dict[str, str] = {}
    for tag in unique_tags:
        quotes = tag_quotes[tag]
        # Deduplicate identical quotes before sending to LLM
        seen = list(dict.fromkeys(quotes))
        best_quotes[tag] = select_best_quote(tag, seen, llm_config)
        print(f"    \u2713 {tag}")

    # Step 3: render markdown
    header_md = render_header_block(profile, transcript_id)
    tables_md = render_code_tables(hits, best_quotes, schema)

    full_md = header_md + "\n" + tables_md
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(full_md, encoding="utf-8")
    print(f"  \u2713 Summary written \u2192 {output_path}")
