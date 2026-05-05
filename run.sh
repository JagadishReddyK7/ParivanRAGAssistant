#!/usr/bin/env bash
# run.sh — One command to set up and launch Parivahan RAG
# Usage: bash run.sh

set -e
echo "=== Parivahan RAG Setup ==="

# 1. Create .env if missing
if [ ! -f .env ]; then
    cp .env.example .env
    echo "[!] Created .env — please add your GEMINI_API_KEY then re-run."
    exit 1
fi

# 2. Install dependencies
echo "Installing dependencies…"
pip install -q -r requirements.txt

# 3. Build knowledge base + index (idempotent)
echo "Building knowledge base…"
python -m src.data_pipeline
python -m src.embeddings

# 4. Launch app
echo "Launching Streamlit…"
streamlit run app.py
