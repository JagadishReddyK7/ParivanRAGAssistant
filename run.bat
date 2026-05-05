@echo off
REM run.bat — Windows one-command setup and launch
REM Usage: run.bat

echo === Parivahan RAG Setup ===

REM 1. Create .env if missing
if not exist .env (
    copy .env.example .env
    echo [!] Created .env — please add your GEMINI_API_KEY then re-run.
    pause
    exit /b 1
)

REM 2. Install dependencies
echo Installing dependencies...
pip install -q -r requirements.txt

REM 3. Build knowledge base + index (idempotent)
echo Building knowledge base...
python -m src.data_pipeline
python -m src.embeddings

REM 4. Launch Streamlit app
echo Launching Streamlit...
streamlit run app.py
