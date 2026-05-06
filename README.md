# Parivahan RAG Assistant

A Retrieval-Augmented Generation (RAG) chatbot designed to answer complex queries about Indian motor vehicle services (Driving Licence, Vehicle RC, Sarathi, Vahan) using official Parivahan documentation.

**🔗 Live Demo:** [https://parivahan-rag-assistant.streamlit.app/](https://parivahan-rag-assistant.streamlit.app/)

## Features
- **Hybrid Search (BM25 + FAISS):** Uses Reciprocal Rank Fusion (RRF) to combine semantic meaning with exact keyword matching (crucial for forms like "Form 4" and acronyms).
- **Dual LLM Support:** Easily toggle between Gemini 2.0 Flash and Groq Llama 3 for generation.
- **Citation Tracking:** In-line citations and dedicated source panels prove the LLM's claims against the retrieved documents.
- **Hindi Localization:** Built-in toggle to answer queries in Hindi (Devanagari script) natively via prompt engineering.

---

## 🛠️ Local Setup Instructions

### Option 1: One-Click Setup (Recommended)
We have included automated launch scripts for both Windows and Mac/Linux. These scripts will automatically install dependencies, build the local vector database, and launch the Streamlit server.

**For Windows:**
```cmd
run.bat
```

**For Mac/Linux:**
```bash
bash run.sh
```
*Note: The script will stop if you don't have a `.env` file. It will create one for you. Simply open it, paste your API keys, and run the script again!*

### Option 2: Manual Setup
If you prefer to run things step-by-step:

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
2. **Configure Environment:**
   Rename `.env.example` to `.env` and add your `GEMINI_API_KEY` or `GROQ_API_KEY`.
3. **Build Knowledge Base:**
   Extracts text from PDFs in `data/pdfs/` and creates chunks.
   ```bash
   python -m src.data_pipeline
   ```
4. **Build Vector Database:**
   Generates embeddings using `all-MiniLM-L6-v2` and creates the FAISS and BM25 indices.
   ```bash
   python -m src.embeddings
   ```
5. **Launch Application:**
   ```bash
   streamlit run app.py
   ```

---

## 📁 Project Structure

Here is a brief overview of what each file in the repository does:

- **`app.py`**: The main Streamlit application file. Handles the UI chat interface, sidebar configurations, and state management.
- **`src/data_pipeline.py`**: Handles downloading PDFs (or reading local ones), extracting text using `PyMuPDF`, and splitting it into manageable 500-word chunks.
- **`src/embeddings.py`**: The core search engine. Embeds text using SentenceTransformers, builds the FAISS vector database, builds the BM25 keyword index, and performs Reciprocal Rank Fusion (RRF) search.
- **`src/rag_pipeline.py`**: Handles prompt engineering and LLM API calls. Formats the retrieved context and sends it to either Gemini or Groq via direct REST API calls.
- **`run.bat` / `run.sh`**: Helper scripts to automate the installation and execution of the entire pipeline.

