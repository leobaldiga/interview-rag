#!/usr/bin/env bash
set -euo pipefail

echo "Creating farmer interview RAG project scaffold in: $(pwd)"

mkdir -p \
  config/prompts \
  data/raw/transcripts \
  data/interim/cleaned \
  data/interim/chunked \
  data/processed/embeddings \
  data/processed/index \
  data/processed/manifests \
  data/metadata \
  outputs/coded_transcripts \
  outputs/chunk_codes \
  outputs/retrieval_logs \
  outputs/summaries \
  outputs/audits \
  logs \
  notebooks \
  src \
  scripts \
  tests

touch \
  README.md \
  requirements.txt \
  .env.example \
  config/settings.yaml \
  config/schema.json \
  config/prompts/system_prompt.txt \
  config/prompts/coding_prompt.txt \
  config/prompts/review_prompt.txt \
  data/metadata/transcript_manifest.csv \
  data/metadata/run_manifest.csv \
  logs/pipeline.log \
  logs/errors.log \
  notebooks/exploration.ipynb \
  src/__init__.py \
  src/ingest.py \
  src/chunking.py \
  src/embeddings.py \
  src/index.py \
  src/retrieve.py \
  src/prompt_builder.py \
  src/code_transcript.py \
  src/merge_codes.py \
  src/export.py \
  src/qa.py \
  src/utils.py \
  scripts/run_ingest.py \
  scripts/run_index.py \
  scripts/run_codebook_pass.py \
  scripts/run_full_coding.py \
  scripts/run_audit_report.py \
  tests/test_chunking.py \
  tests/test_prompt_builder.py \
  tests/test_retrieval.py

cat > .gitignore <<'GITIGNORE'
# Python
__pycache__/
*.py[cod]
*.pyo
.venv/
venv/
.env

# Project outputs
data/processed/embeddings/*
data/processed/index/*
outputs/*
logs/*.log

# Keep folder structure
!outputs/.gitkeep
!outputs/coded_transcripts/.gitkeep
!outputs/chunk_codes/.gitkeep
!outputs/retrieval_logs/.gitkeep
!outputs/summaries/.gitkeep
!outputs/audits/.gitkeep

# Jupyter
.ipynb_checkpoints/
GITIGNORE

touch \
  outputs/.gitkeep \
  outputs/coded_transcripts/.gitkeep \
  outputs/chunk_codes/.gitkeep \
  outputs/retrieval_logs/.gitkeep \
  outputs/summaries/.gitkeep \
  outputs/audits/.gitkeep

echo "Done."
echo "Project scaffold created successfully."
