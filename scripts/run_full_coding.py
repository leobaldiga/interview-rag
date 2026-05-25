from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

import yaml

from src.chunking import chunk_transcript
from src.prompt_builder import build_coding_prompt
from src.code_transcript import code_with_llama


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_jsonl(records: list[dict], path: Path) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def save_markdown(text: str, path: Path) -> None:
    ensure_dir(path.parent)
    path.write_text(text, encoding="utf-8")


def save_run_manifest(rows: list[dict], path: Path) -> None:
    ensure_dir(path.parent)
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def merge_chunk_outputs(coded_chunks: list[dict]) -> str:
    parts = []
    for row in coded_chunks:
        parts.append(
            f"<!-- {row['chunk_id']} | chars {row['start_char']}-{row['end_char']} -->\n\n"
            f"{row['coded_text'].strip()}"
        )
    return "\n\n---\n\n".join(parts).strip()


def make_output_stem(transcript_id: str, date_suffix: str) -> str:
    """Return the output file stem with _clean replaced by _coded_{MMDDYYYY}.

    Examples
    --------
    >>> make_output_stem("20260331_ARINT006_clean", "05252026")
    '20260331_ARINT006_coded_05252026'
    >>> make_output_stem("20260331_ARINT006", "05252026")
    '20260331_ARINT006_coded_05252026'
    """
    base = transcript_id
    if base.endswith("_clean"):
        base = base[: -len("_clean")]
    return f"{base}_coded_{date_suffix}"


def main() -> None:
    settings = yaml.safe_load(Path("config/settings.yaml").read_text(encoding="utf-8"))

    raw_dir = Path(settings["paths"]["raw_transcripts_dir"])
    chunk_codes_dir = Path(settings["paths"]["chunk_codes_dir"])
    coded_transcripts_dir = Path(settings["paths"]["coded_transcripts_dir"])
    manifest_dir = Path(settings["paths"]["manifest_dir"])

    parser = argparse.ArgumentParser(description="Code transcript txt files with llama.cpp")
    parser.add_argument(
        "--file",
        dest="target_files",
        nargs="+",
        help=(
            "One or more transcript filenames to process from "
            f"{raw_dir} (e.g. 20260311_ARINT009_clean.txt)"
        ),
    )
    parser.add_argument(
        "--schema",
        dest="schema_path",
        default=None,
        help=(
            "Path to a schema JSON file to use for this run. "
            "Overrides the schema path in config/settings.yaml."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show which transcripts would be processed, but do not call the model or write outputs.",
    )
    args = parser.parse_args()

    schema_path = (
        Path(args.schema_path) if args.schema_path else Path(settings["paths"]["schema_file"])
    )

    if not schema_path.exists():
        print(f"Schema file not found: {schema_path}")
        return

    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    print(f"Using schema: {schema_path}")

    ensure_dir(chunk_codes_dir)
    ensure_dir(coded_transcripts_dir)
    ensure_dir(manifest_dir)

    if args.target_files:
        txt_files: list[Path] = []
        for name in args.target_files:
            target_path = raw_dir / name
            if not target_path.is_file():
                print(f"Requested file not found or not a file: {target_path}")
                continue
            txt_files.append(target_path)

        if not txt_files:
            print("No valid --file targets found, nothing to do.")
            return
    else:
        txt_files = sorted(raw_dir.glob("*.txt"))

    if not txt_files:
        print(f"No .txt transcripts found in {raw_dir}")
        return

    if args.dry_run:
        print(f"Dry run: using schema → {schema_path}")
        print("Dry run: would process the following transcripts (in order):")
        for p in txt_files:
            print(f"  - {p.name}")
        return

    now = datetime.now()
    run_id = now.strftime("%Y-%m-%d_%H%M%S")
    date_suffix = now.strftime("%m%d%Y")
    run_rows = []

    for txt_file in txt_files:
        transcript_id = txt_file.stem
        output_stem = make_output_stem(transcript_id, date_suffix)
        print(f"\nProcessing: {transcript_id}  →  {output_stem}")

        raw_text = txt_file.read_text(encoding="utf-8")
        chunks = chunk_transcript(
            raw_text=raw_text,
            transcript_id=transcript_id,
            chunk_config=settings["chunking"],
        )

        coded_chunks = []
        total_chunks = len(chunks)
        jsonl_path = chunk_codes_dir / f"{output_stem}.jsonl"

        if jsonl_path.exists():
            jsonl_path.unlink()

        for i, chunk in enumerate(chunks, start=1):
            print(
                f"  Coding chunk {i:04d}/{total_chunks:04d} | "
                f"{chunk['chunk_id']} | {len(chunk['text'])} chars"
            )

            messages = build_coding_prompt(
                chunk=chunk["text"],
                transcript_id=chunk["transcript_id"],
                chunk_id=chunk["chunk_id"],
                schema=schema,
                settings=settings,
            )

            coded_text = code_with_llama(
                messages=messages,
                llm_config=settings["llm"],
            )

            print(f"\n--- coded output: {chunk['chunk_id']} ---")
            print(coded_text)
            print("--- end coded output ---\n")

            record = {
                "run_id": run_id,
                "transcript_id": chunk["transcript_id"],
                "chunk_id": chunk["chunk_id"],
                "chunk_number": i,
                "total_chunks": total_chunks,
                "start_char": chunk["start_char"],
                "end_char": chunk["end_char"],
                "source_text": chunk["text"],
                "coded_text": coded_text,
            }

            coded_chunks.append(record)

            with jsonl_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        merged_md = merge_chunk_outputs(coded_chunks)
        save_markdown(merged_md, coded_transcripts_dir / f"{output_stem}.md")

        run_rows.append(
            {
                "run_id": run_id,
                "schema": schema_path.name,
                "transcript_id": transcript_id,
                "output_stem": output_stem,
                "n_chunks": len(chunks),
                "chunk_output_file": f"{output_stem}.jsonl",
                "merged_output_file": f"{output_stem}.md",
            }
        )

    manifest_path = manifest_dir / f"run_{run_id}.csv"
    save_run_manifest(run_rows, manifest_path)
    print(f"\nDone. Run manifest saved to {manifest_path}")

if __name__ == "__main__":
    main()
