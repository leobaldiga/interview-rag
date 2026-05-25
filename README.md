# interview-rag

A local Python pipeline for coding long farmer interview transcripts with a fixed qualitative coding schema using `llama.cpp` and chunked processing.

## What it does

- Reads plain-text interview transcripts from `data/raw/transcripts/`
- Splits each transcript into overlapping chunks
- Sends each chunk to a local `llama.cpp` OpenAI-compatible chat endpoint
- Applies a fixed P-S-T qualitative coding schema
- Saves chunk-level outputs as JSONL (filename includes date of run)
- Saves merged transcript outputs as Markdown
- Writes a run manifest for tracking each coding run
- Generates per-interview summary documents with farm profile headers and per-code verbatim quote tables

## Current status

Core coding pipeline is working. Summary generation is newly implemented.

Current implemented components:

- Project scaffold and directory structure
- `settings.yaml` configuration file
- `schema.json` coding schema (7 primary themes, 24 subthemes)
- Chunking logic for long transcripts
- Prompt builder for schema-aware coding prompts
- Local `llama-server` API integration
- Incremental JSONL writing after each completed chunk
- Merged Markdown export per transcript with date-stamped filenames
- Per-interview summary generation with LLM-extracted farm profile and best-quote tables

Planned improvements:

- Better speaker-aware chunking
- Retry / resume logic for interrupted runs
- Output validation against the schema
- Embeddings + retrieval for a fuller RAG workflow
- Better audit reporting and summary statistics

## Project structure

```text
interview-rag/
├── config/
│   ├── prompts/
│   ├── schema.json
│   └── settings.yaml
├── data/
│   ├── raw/transcripts/
│   ├── interim/
│   ├── processed/
│   └── metadata/
├── logs/
├── notebooks/
├── outputs/
│   ├── audits/
│   ├── chunk_codes/
│   ├── coded_transcripts/
│   ├── retrieval_logs/
│   └── summaries/
├── scripts/
│   ├── run_full_coding.py
│   └── generate_summaries.py
├── src/
│   ├── chunking.py
│   ├── code_transcript.py
│   ├── prompt_builder.py
│   └── summarize.py
└── tests/
```

## Requirements

- Python 3.10+
- A local virtual environment (`.venv` recommended)
- A running `llama.cpp` server
- Plain-text transcript files (`.txt`)

Python packages currently used:

- `requests`
- `pyyaml`

Install inside a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install requests pyyaml
```

## Running llama.cpp

A typical local server command looks like this:
```bash
docker run -d \
  --name llama-server \
  --gpus all \
  -p 8080:8080 \
  -v ~/models:/models \
  ghcr.io/ggml-org/llama.cpp:server-cuda \
  -m /models/gemma-4-E4B-it-Q8_0.gguf \
  --mmproj /models/mmproj-BF16.gguf \
  --host 0.0.0.0 \
  --port 8080 \
  -ngl 99 \
  -c 128000 \
  --cache-prompt
```

CPU only fallback (VERY SLOW)
```
docker run -d \
  --name llama-server-cpu \
  -p 8080:8080 \
  -v ~/models:/models \
  ghcr.io/ggml-org/llama.cpp:server \
  -m /models/gemma-4-E4B-it-Q8_0.gguf \
  --host 0.0.0.0 \
  --port 8080 \
  -c 8192 \
  --cache-prompt
```

Adjust the model paths and parameters for your hardware and model choice.

## Configuration

The two most important config files are:

### `config/settings.yaml`

Defines:

- Input and output paths
- LLM endpoint settings
- Chunking parameters
- Retrieval settings for later RAG extensions
- Coding options

### `config/schema.json`

Contains the qualitative coding schema used by the prompt builder and coding pipeline.

The schema uses a three-level P-S-T hierarchy:

- **P (Primary theme)** — top-level thematic domain
- **S (Subtheme)** — mid-level grouping within a primary theme
- **T (Tertiary tag)** — specific coded concept, inserted inline in the transcript

Tags are written in the format:
```
<P_PRIMARY:S_SECONDARY:T:tag-text>
```

Current primary themes:

| Code | Theme |
|------|-------|
| `P_AOP` | Agricultural Operation & Practice |
| `P_SH` | Soil Health |
| `P_MSK` | Management Systems and Knowledge |
| `P_RRM` | Resilience and Risk Management |
| `P_CEF` | Climate and External Factors |
| `P_FOSC` | Future Outlook and System Change |
| `P_ENP` | Energy Policy and Transition |

`P_ENP` subthemes:

| Code | Subtheme |
|------|----------|
| `S_EFC` | On-Farm Energy Use and Costs |
| `S_REA` | Renewable Energy Adoption |
| `S_EPR` | Energy Policy and Regulatory Environment |

## Input data

Place transcript files here:

```text
data/raw/transcripts/
```

Expected input format:

- Plain `.txt` files
- One transcript per file
- Filename becomes `transcript_id`

Recommended naming convention:

```text
data/raw/transcripts/YYYYMMDD_ARINTXXX_clean.txt
```

Example:

```text
data/raw/transcripts/20260331_ARINT006_clean.txt
```

## Running the pipeline

### Step 1: Code transcripts

From the project root:

```bash
source .venv/bin/activate
PYTHONPATH=. python scripts/run_full_coding.py
```

To process a specific file:

```bash
PYTHONPATH=. python scripts/run_full_coding.py --file 20260331_ARINT006_clean.txt
```

To preview which files would be processed without running the model:

```bash
PYTHONPATH=. python scripts/run_full_coding.py --dry-run
```

Example terminal output:

```text
Processing: 20260331_ARINT006_clean  →  20260331_ARINT006_coded_05252026
  Coding chunk 0001/0042 | 20260331_ARINT006_clean__00000 | 2082 chars
    ↳ 14.3s | prompt=1204 tokens | completion=892 tokens | finish=stop
```

### Step 2: Generate summaries

After coding is complete, generate per-interview summary documents:

```bash
PYTHONPATH=. python scripts/generate_summaries.py
```

To summarize a specific coded JSONL file:

```bash
PYTHONPATH=. python scripts/generate_summaries.py --file 20260331_ARINT006_coded_05252026.jsonl
```

The summary script:

1. Parses all tagged speaker turns from the chunk-level JSONL
2. Calls the LLM to extract a farm profile header (demographics, rotation, tillage, cover crops, tenure, etc.) from the raw transcript
3. Calls the LLM once per unique tag to select the single most substantive verbatim quote from all candidate passages
4. Renders a Markdown file with a farm profile table and per-subtheme quote tables

## Outputs

### Output file naming

All output files include the date the coding run was completed:

```text
Input:  20260331_ARINT006_clean.txt
Output: 20260331_ARINT006_coded_05252026.md
        20260331_ARINT006_coded_05252026.jsonl
        20260331_ARINT006_coded_05252026_summary.md
```

The date suffix format is `MMDDYYYY`.

### Chunk-level JSONL

Written to:

```text
outputs/chunk_codes/
```

Each line contains a JSON record with fields:

- `run_id`
- `transcript_id`
- `chunk_id`
- `chunk_number`
- `total_chunks`
- `start_char`
- `end_char`
- `source_text`
- `coded_text`

### Merged Markdown transcript

Written to:

```text
outputs/coded_transcripts/
```

Chunk outputs stitched into a single Markdown file separated by `---` dividers, each chunk preceded by a comment with its chunk ID and character offsets.

### Interview summary

Written to:

```text
outputs/summaries/
```

Each summary contains:

- **Farm profile table** — LLM-extracted fields including farmer ID, location, farm size, land tenure, years farming, primary crops, crop rotation, cover cropping, tillage system, irrigation, and notable context
- **Coded theme tables** — one section per primary theme, one table per subtheme, with the single best verbatim quote selected per tag

### Run manifest

Written to:

```text
data/processed/manifests/
```

Tracks which transcripts were processed in a run, what schema was used, and what output files were produced. Fields include `run_id`, `schema`, `transcript_id`, `output_stem`, `n_chunks`, `chunk_output_file`, and `merged_output_file`.

## Development notes

This project is intentionally being built in stages:

1. Chunk long transcripts reliably
2. Code each chunk against the schema
3. Save outputs safely after each chunk
4. Generate per-interview summaries with farm profile and best-quote tables
5. Add validation and resumability
6. Add embeddings and retrieval for a fuller RAG workflow

## Next priorities

- Add schema validation for inserted tags
- Add try/except handling so one failed chunk does not kill the entire transcript
- Add resume mode from existing JSONL files
- Improve chunking around speaker turns
- Add embedding-based retrieval to move from chunked coding to fuller local RAG
