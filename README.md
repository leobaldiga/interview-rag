# interview-rag

A local Python pipeline for coding long farmer interview transcripts with a fixed qualitative coding schema using `llama.cpp` and chunked processing.

## What it does

- Reads plain-text interview transcripts from `data/raw/transcripts/`
- Splits each transcript into overlapping chunks
- Sends each chunk to a local `llama.cpp` OpenAI-compatible chat endpoint
- Applies a fixed P-S-T qualitative coding schema
- Saves chunk-level outputs as JSONL
- Saves merged transcript outputs as Markdown
- Writes a run manifest for tracking each coding run

## Current status

This is an early but working local pipeline.

Current implemented components:

- Project scaffold and directory structure
- `settings.yaml` configuration file
- `schema.json` coding schema file
- Chunking logic for long transcripts
- Prompt builder for schema-aware coding prompts
- Local `llama-server` API integration
- Incremental JSONL writing after each completed chunk
- Merged Markdown export for each transcript

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
│   └── run_full_coding.py
├── src/
│   ├── chunking.py
│   ├── code_transcript.py
│   └── prompt_builder.py
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

## Input data

Place transcript files here:

```text
data/raw/transcripts/
```

Expected input format:

- Plain `.txt` files
- One transcript per file
- Filename becomes `transcript_id`

Example:

```text
data/raw/transcripts/20260401_ARINT001.txt
```

## Running the pipeline

From the project root:

```bash
source .venv/bin/activate
PYTHONPATH=. python scripts/run_full_coding.py
```

Example terminal output:

```text
Processing: 20260401_ARINT001
  Coding chunk 0001/0042 | 20260401_ARINT001__00000 | 2082 chars
```

The script currently:

- Prints progress with numbered chunks
- Prints coded output after each chunk finishes
- Appends each finished chunk immediately to the transcript JSONL file

## Outputs

### Chunk-level JSONL

Written to:

```text
outputs/chunk_codes/
```

Each line contains a JSON record with fields such as:

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

This contains the chunk outputs stitched together into a single Markdown file for easier review.

### Run manifest

Written to:

```text
data/processed/manifests/
```

This tracks which transcripts were processed in a run and what files were produced.

## Development notes

This project is intentionally being built in stages:

1. Chunk long transcripts reliably
2. Code each chunk against the schema
3. Save outputs safely after each chunk
4. Add validation and resumability
5. Add embeddings and retrieval for a fuller RAG workflow

That staged approach makes debugging easier and reduces the risk of losing long-running local jobs.

## Next priorities

- Add schema validation for inserted tags
- Add try/except handling so one failed chunk does not kill the entire transcript
- Add resume mode from existing JSONL files
- Improve chunking around speaker turns
- Add embedding-based retrieval to move from chunked coding to fuller local RAG