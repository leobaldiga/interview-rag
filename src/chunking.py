from __future__ import annotations

import re
from typing import Dict, List


def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_paragraphs(text: str) -> List[str]:
    parts = re.split(r"\n\s*\n", text)
    return [p.strip() for p in parts if p.strip()]


def chunk_transcript(
    raw_text: str,
    transcript_id: str,
    chunk_config: Dict
) -> List[Dict]:
    text = normalize_text(raw_text)
    paragraphs = split_paragraphs(text)

    target_chars = int(chunk_config.get("target_chars", 3500))
    overlap_chars = int(chunk_config.get("overlap_chars", 350))
    min_chunk_chars = int(chunk_config.get("min_chunk_chars", 1600))
    max_chunk_chars = int(chunk_config.get("max_chunk_chars", 4200))

    chunks: List[Dict] = []
    current_parts: List[str] = []
    current_text = ""
    chunk_start = 0

    for para in paragraphs:
        candidate = f"{current_text}\n\n{para}".strip() if current_text else para

        if len(candidate) <= target_chars:
            current_parts.append(para)
            current_text = candidate
            continue

        if current_text and len(current_text) >= min_chunk_chars:
            chunk_end = chunk_start + len(current_text)
            chunk_id = f"{transcript_id}__{len(chunks):05d}"
            chunks.append(
                {
                    "transcript_id": transcript_id,
                    "chunk_id": chunk_id,
                    "text": current_text[:max_chunk_chars],
                    "start_char": chunk_start,
                    "end_char": chunk_end,
                }
            )

            overlap_text = current_text[-overlap_chars:] if overlap_chars > 0 else ""
            current_text = f"{overlap_text}\n\n{para}".strip() if overlap_text else para
            current_parts = [current_text]
            chunk_start = max(0, chunk_end - len(overlap_text))
        else:
            forced = candidate[:max_chunk_chars]
            chunk_end = chunk_start + len(forced)
            chunk_id = f"{transcript_id}__{len(chunks):05d}"
            chunks.append(
                {
                    "transcript_id": transcript_id,
                    "chunk_id": chunk_id,
                    "text": forced,
                    "start_char": chunk_start,
                    "end_char": chunk_end,
                }
            )
            remainder = candidate[max_chunk_chars:].strip()
            current_text = remainder
            current_parts = [remainder] if remainder else []
            chunk_start = chunk_end

    if current_text.strip():
        chunk_end = chunk_start + len(current_text)
        chunk_id = f"{transcript_id}__{len(chunks):05d}"
        chunks.append(
            {
                "transcript_id": transcript_id,
                "chunk_id": chunk_id,
                "text": current_text[:max_chunk_chars],
                "start_char": chunk_start,
                "end_char": chunk_end,
            }
        )

    return chunks
