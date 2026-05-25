from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import yaml

from src.summarize import generate_summary


def main() -> None:
    settings = yaml.safe_load(Path("config/settings.yaml").read_text(encoding="utf-8"))

    chunk_codes_dir = Path(settings["paths"]["chunk_codes_dir"])
    raw_transcripts_dir = Path(settings["paths"]["raw_transcripts_dir"])
    summaries_dir = Path(settings["paths"]["summaries_dir"])
    schema_path = Path(settings["paths"]["schema_file"])

    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    llm_config = settings["llm"]

    parser = argparse.ArgumentParser(
        description="Generate per-interview markdown summary tables from coded JSONL outputs."
    )
    parser.add_argument(
        "--file",
        dest="target_files",
        nargs="+",
        help="One or more .jsonl filenames from outputs/chunk_codes/ to summarize.",
    )
    args = parser.parse_args()

    if args.target_files:
        jsonl_files = []
        for name in args.target_files:
            p = chunk_codes_dir / name
            if not p.is_file():
                print(f"  \u2717 Not found: {p}")
            else:
                jsonl_files.append(p)
    else:
        jsonl_files = sorted(chunk_codes_dir.glob("*.jsonl"))

    if not jsonl_files:
        print(f"No .jsonl files found in {chunk_codes_dir}")
        return

    for jsonl_path in jsonl_files:
        stem = jsonl_path.stem
        transcript_id = stem

        # Resolve matching raw transcript
        # Coded stem pattern: YYYYMMDD_ARINTXXX_coded_MMDDYYYY
        # Raw file pattern:   YYYYMMDD_ARINTXXX_clean.txt
        raw_candidate = re.sub(r"_coded_\d{8}$", "_clean", stem)
        raw_path = raw_transcripts_dir / f"{raw_candidate}.txt"
        if not raw_path.exists():
            # Fallback: try without _clean suffix
            raw_path = raw_transcripts_dir / f"{re.sub(r'_coded_\\d{8}$', '', stem)}.txt"

        output_path = summaries_dir / f"{stem}_summary.md"

        print(f"\nSummarizing: {stem}")
        if raw_path.exists():
            print(f"  Raw transcript: {raw_path.name}")
        else:
            print(f"  \u26a0\ufe0f  Raw transcript not found \u2014 header will be empty.")

        generate_summary(
            jsonl_path=jsonl_path,
            raw_transcript_path=raw_path,
            output_path=output_path,
            schema=schema,
            llm_config=llm_config,
            transcript_id=transcript_id,
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
