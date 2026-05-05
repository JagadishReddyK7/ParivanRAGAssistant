# Parivahan RAG Chatbot — Viva & Interview Prep Guide

---

## 1. SYSTEM OVERVIEW

### End-to-End Flow

```
PDFs (Parivahan docs)
    │
    ▼  [data_pipeline.py]
Text Extraction (PyMuPDF)
    │  Each page → cleaned text → split into 500-word chunks with 50-word overlap
    ▼
chunks.json  ──────────────────────────────────────────────────┐
    │                                                           │
    ▼  [embeddings.py]                                         │
SentenceTransformer (all-MiniLM-L6-v2)                        │
    │  Each chunk → 384-dim float32 vector                     │
    ▼                                                          │
FAISS Index (IndexFlatIP)  +  metadata.pkl                    │
    │                                                          │
    ▼  [app.py — user types query]                            │
Query → same embedding model → 384-dim query vector            │
    │                                                          │
    ▼  [embeddings.search()]                                  │
FAISS cosine similarity search → top-K chunks retrieved ◄──────┘
    │
    ▼  [rag_pipeline.py]
Prompt = System instructions + retrieved chunks + user question
    │
    ▼
LLM API (Gemini 1.5 Flash or Groq Llama3)
    │
    ▼  [app.py — Streamlit UI]
Answer displayed + Citation box [Source - page X]
```

**Simple explanation**: Think of it like a smart open-book exam.
The PDF is the book. Embeddings index the book by meaning. When you ask a question, the system finds the most relevant pages, hands them to the LLM, and says "answer using only this." The LLM writes the answer and cites the pages.

---

## 2. COMPONENT-WISE BREAKDOWN

### `data_pipeline.py` — The Librarian

**What it does**: Downloads PDFs, extracts text, splits into chunks, saves to JSON.

**Why needed**: LLMs have a context window limit (~8K–32K tokens). You cannot paste an entire PDF. Chunking breaks it into retrievable pieces.

**Key function**:

```python
def parse_pdf(pdf_path, chunk_size=500):
    doc = fitz.open(str(pdf_path))          # PyMuPDF opens PDF
    for page_num, page in enumerate(doc):
        text = page.get_text("text")         # extract raw text
        text = re.sub(r'\s+', ' ', text)     # collapse whitespace/newlines
        words = text.split()
        step = chunk_size - 50               # 50-word overlap
        for i, start in enumerate(range(0, len(words), step)):
            chunk_text = " ".join(words[start: start + chunk_size])
            # store: text, source name, page number, chunk_id
```

**Why 50-word overlap?** A sentence spanning a chunk boundary won't be lost. Both adjacent chunks contain it.

**Fallback**: If PDF download fails (Parivahan URLs are often 404), 6 hard-coded knowledge chunks are used — making the system demo-ready offline.

**Alternatives**: `pdfplumber` (better for tables), `unstructured` (handles scanned/OCR PDFs), LangChain `RecursiveCharacterTextSplitter` (sentence-aware).

---

### `embeddings.py` — The Indexer

**What it does**: Converts text chunks to vectors, builds a FAISS index, provides search.

**Key functions**:

```python
def embed_chunks(model, chunks):
    embeddings = model.encode(
        texts,
        normalize_embeddings=True   # unit-length → cosine sim = dot product
    )
    return embeddings.astype("float32")  # FAISS requires float32

def build_index(embeddings):
    index = faiss.IndexFlatIP(dim)  # IP = Inner Product (= cosine after normalization)
    index.add(embeddings)           # all vectors stored in RAM
    return index

def search(query, model, index, meta, top_k=5):
    qvec = model.encode([query], normalize_embeddings=True)
    scores, ids = index.search(qvec, top_k)
    return [meta[idx] for idx in ids[0]]
```

**Why `IndexFlatIP` not `IndexFlatL2`?**
After L2-normalization, inner product = cosine similarity. Cosine measures directional similarity (topic matching), not magnitude — better for text.

**Alternatives**: `IndexIVFFlat` (approximate, faster at 1M+ vectors), ChromaDB (persistent), Pinecone/Weaviate (cloud-hosted).

---

### `rag_pipeline.py` — The Brain

**What it does**: Takes retrieved chunks, builds a prompt, calls the LLM, returns the answer.

**Key design**:

```python
SYSTEM_PROMPT = """You are a knowledgeable assistant...
Answer ONLY from the provided context.
If the context doesn't contain the answer, say "I don't have enough information."
Always cite sources as [SOURCE - page X]
{hindi_instruction}"""
```

**Why "Answer ONLY from context"?** This is the core anti-hallucination directive. Without it, the LLM uses its training data which may be outdated or wrong.

**Citation injection**: Citation labels are prepended INSIDE each chunk before sending to the LLM:
```
[DL_Procedure - page 1]
Step 1: Obtain Learner's Licence...
```
The LLM sees and reproduces these labels in its output naturally.

**Hindi mode**: Appends `"Respond in Hindi (Devanagari script)"` to the system prompt. No separate translation model needed.

---

### `app.py` — The Interface

**Key patterns**:

```python
@st.cache_resource          # runs ONCE, cached for entire session
def init_rag():             # loads model + FAISS index on startup
    ...

st.session_state.messages  # persists chat history across Streamlit reruns
st.chat_input(...)         # native chat widget
```

Defensive JSON parsing — if `chunks.json` is empty/corrupt (from a failed write), it rebuilds automatically instead of crashing.

---

## 3. CORE CONCEPTS

### RAG (Retrieval-Augmented Generation)
Standard LLMs hallucinate and have stale training data. RAG solves this:
1. **Retrieve** — find relevant documents at query time from your own data
2. **Augment** — inject them into the LLM prompt as context
3. **Generate** — LLM answers grounded in real documents

RAG = Search Engine + LLM, combined.

### Embeddings (all-MiniLM-L6-v2)
A vector of numbers that captures the *meaning* of text in a mathematical space.
"How to get DL?" and "Procedure for driving licence" → similar vectors → high cosine similarity.
MiniLM-L6-v2: 6 transformer layers, 384-dim output, 80MB size, runs fast on CPU.

### FAISS Vector Search
FAISS stores all chunk vectors in RAM. At query time:
1. Query → 384-dim vector
2. Compare against all stored vectors
3. Return top-K closest by cosine similarity

`IndexFlatIP` = exact brute-force search. Correct for small datasets. For millions of vectors, use approximate indices (IVF, HNSW).

### Chunking Strategy
- **Size**: 500 words ≈ 1–2 paragraphs. Large enough for context, small enough for precision.
- **Overlap**: 50 words prevents information loss at boundaries.
- **Limitation**: Fixed-size chunking can split mid-sentence. Semantic chunking is better.

### Prompt Engineering
4 key directives in the system prompt:
1. **Role** ("specialist in Indian motor vehicle services")
2. **Grounding** ("Answer ONLY from provided context")
3. **Fallback** ("say I don't know if context is insufficient")
4. **Citation format** ("cite as [SOURCE - page X]")

---

## 4. DESIGN DECISIONS

| Decision | Why | Alternative |
|---|---|---|
| MiniLM-L6-v2 | Tiny (80MB), CPU-fast, 90% accuracy of large models | `bge-large`, OpenAI `ada-002` (paid) |
| FAISS | Zero setup, in-memory, no server needed | ChromaDB (persistent), Pinecone (cloud) |
| Gemini 1.5 Flash | Free tier, 1M context window, multilingual | GPT-4o (paid), Claude (paid) |
| Groq Llama3 | Free, fastest inference (LPU hardware) | Ollama (fully local, no internet) |
| Streamlit | 10 lines for full chat UI | Gradio, FastAPI + React |
| PyMuPDF | Fastest PDF parser | pdfplumber (better tables), pypdf (simpler) |
| CPU only | No GPU needed, runs on any laptop | GPU gives 10–50x speedup for embedding |

---

## 5. PROBLEMS FACED

### P1: PDF URLs Return 404
**Why**: Parivahan has unstable file paths, frequent restructuring.
**Code handles it**: `try/except` in `download_pdf()` → falls back to 6 hard-coded chunks.
**Better**: Scrape PDF links dynamically with Playwright; use Wayback Machine archives.

### P2: PDF Parsing Noise
**Why**: PDFs contain headers, footers, page numbers, watermarks in extracted text.
**Code handles it**: `re.sub(r'\s+', ' ', text)` normalizes whitespace. Chunks < 80 chars skipped.
**Better**: Regex to strip page numbers. pdfplumber for structured extraction.

### P3: Irrelevant Retrieval
**Why**: Semantic similarity doesn't always equal topical relevance for short queries.
**Code handles it**: Top-K=5 (adjustable). System prompt says to decline if context is insufficient.
**Better**: Similarity score threshold (reject chunks < 0.4). Cross-encoder re-ranking.

### P4: Hallucination
**Why**: LLMs generate plausible-sounding but wrong information when context is ambiguous.
**Code handles it**: System prompt: "Do NOT hallucinate fees, URLs, or form numbers."
**Better**: `temperature=0`, post-generation fact checking, confidence thresholding.

### P5: Unicode Error on Windows
**Why**: Windows default encoding `cp1252` can't handle `→`, `✓`, Devanagari.
**Fix**: `write_text(..., encoding="utf-8")` and `read_text(..., encoding="utf-8")`.

### P6: First-load Latency
**Why**: Loading MiniLM + FAISS takes 2–5 seconds.
**Code handles it**: `@st.cache_resource` — runs once, cached in memory for the session.

---

## 6. IMPROVEMENTS

### Accuracy
- **Re-ranking**: Retrieve top-10 with FAISS, re-rank with a cross-encoder, use top-3
- **Hybrid search**: BM25 (keyword) + FAISS (semantic) combined
- **Semantic chunking**: Split at paragraph/sentence boundaries, not word count
- **HyDE**: Generate a hypothetical answer first, embed it, search with that vector

### Scale
- `IndexIVFFlat` for approximate search at 1M+ vectors
- ChromaDB/Weaviate for persistent storage (no re-indexing on restart)
- Redis cache for repeated query responses

### Production
- FastAPI backend (not Streamlit — single-threaded)
- Async LLM calls with streaming (`st.write_stream`)
- Auth layer, rate limiting, query logging
- Docker already provided; add `docker-compose` with health checks

---

## 7. INTERVIEW Q&A

### Basic

**Q: What is RAG?**
RAG (Retrieval-Augmented Generation) retrieves relevant documents from a knowledge base at query time and injects them as context into the LLM prompt. This grounds answers in real data, reduces hallucination, and allows the LLM to answer about documents it was never trained on.

**Q: What is an embedding?**
A dense vector of numbers representing the semantic meaning of text. Similar meanings → similar vectors → high cosine similarity. MiniLM produces 384-dimensional vectors.

**Q: What is FAISS?**
Facebook AI Similarity Search — a library for fast nearest-neighbor search over dense vectors. Stores all document vectors and returns the closest matches to a query vector.

---

### Intermediate

**Q: Why MiniLM over larger embedding models?**
MiniLM-L6-v2 is knowledge-distilled from a larger teacher model into 6 layers. It achieves ~90% of large model performance on semantic similarity benchmarks while being 10x smaller and running in milliseconds on CPU. For a domain-specific government services chatbot with limited data, the quality difference is negligible.

**Q: Why `IndexFlatIP` and not `IndexFlatL2`?**
Embeddings are L2-normalized to unit length. For unit vectors, inner product equals cosine similarity. Cosine measures directional similarity (topic matching) independent of magnitude, which is better for text than Euclidean distance.

**Q: Why does overlap matter in chunking?**
A sentence spanning two consecutive chunks would be missed during retrieval if no overlap exists — neither chunk contains it fully. 50-word overlap ensures both adjacent chunks contain boundary-spanning content.

**Q: Why put citation labels inside the context?**
The LLM sees `[DL_Procedure - page 1]` directly in its input. It then naturally reproduces those labels in its output. This is more reliable than asking the LLM to generate citations from memory, which can hallucinate page numbers.

---

### Advanced

**Q: How would you reduce hallucination further?**
1. `temperature=0` for deterministic output
2. Similarity score threshold — if max retrieved score < 0.4, return "I don't know" without calling LLM
3. Post-generation verification — extract factual claims and check against retrieved chunks
4. Use smaller, focused prompts with fewer chunks
5. Fine-tune the LLM on domain-specific Q&A pairs

**Q: How would you handle a multi-topic query (DL + RC)?**
1. Increase top-K and let the LLM select relevant parts
2. Query decomposition — split into sub-queries, retrieve independently, merge results
3. Metadata filtering — tag chunks by topic (DL/RC/IDP) and route accordingly

**Q: What if the answer is not in the knowledge base?**
The system prompt instructs the LLM to say "I don't have enough information." However, if retrieved chunks are topically similar but don't contain the exact answer, the LLM may still hallucinate. Better: check if `max(scores) < threshold` and short-circuit before LLM call.

**Q: How would you scale to 10 million chunks?**
1. Replace `IndexFlatIP` (O(n) exact) with `IndexIVFFlat` or `IndexHNSWFlat` (O(log n) approximate)
2. Use Pinecone/Weaviate with metadata filtering (search only DL chunks for DL queries)
3. Distributed embedding generation on GPU cluster
4. Cache embeddings for repeated queries (Redis)

---

### System Design

**Q: Design a production RAG system for 100 concurrent users.**
- **Embedding**: Pre-computed offline, stored in Pinecone with metadata
- **Query layer**: FastAPI async — query embedding + FAISS search in parallel
- **LLM**: Async calls with connection pooling, retry + exponential backoff
- **Caching**: Redis TTL cache for repeated identical queries
- **Frontend**: Next.js with SSE for streaming responses
- **Monitoring**: Log all queries + chunks + answers to PostgreSQL; compute RAGAS metrics periodically
- **Cost control**: Rate limit per user, cache LLM responses for 1 hour

---

## 8. LEARNING ROADMAP

### Must-Know (Before Viva)
1. Transformer architecture — attention, BERT
2. Sentence embeddings — how MiniLM produces fixed-size vectors
3. Cosine similarity vs Euclidean distance
4. FAISS index types — Flat, IVF, HNSW
5. Prompt engineering — system vs user prompt, temperature, max_tokens
6. RAG vs Fine-tuning — when to use each

### Good-to-Know
7. BM25 — keyword retrieval for hybrid search
8. Cross-encoders — re-ranking retrieved results
9. LangChain/LlamaIndex — RAG frameworks (understand what they abstract)
10. Vector DB comparison — FAISS vs ChromaDB vs Pinecone vs Weaviate
11. Chunking strategies — fixed-size, sentence, semantic, parent-child

### Advanced
12. HyDE — Hypothetical Document Embeddings
13. RAPTOR — recursive abstractive processing for long documents
14. RAGAS — evaluation metrics (faithfulness, answer relevance, context recall)
15. Quantization — INT8/INT4 models for faster CPU inference
16. RLHF/DPO — how LLMs are aligned for instruction following

---

## 9. VIVA SPEECH (2–3 minutes)

> "Our project is a RAG-based chatbot that answers questions about Indian driving licences and vehicle registration using official Parivahan government documents.
>
> The system has two phases. In the **offline indexing phase**, we download PDFs, extract text using PyMuPDF, and split each page into 500-word overlapping chunks. Each chunk is converted to a 384-dimensional semantic vector using the `all-MiniLM-L6-v2` sentence transformer — lightweight enough to run fully on CPU. These vectors are stored in a FAISS index with their metadata.
>
> In the **online query phase**, when a user types a question, it's embedded using the same model. FAISS computes cosine similarity between the query vector and all stored chunk vectors and returns the top 5 most relevant chunks. These chunks — with their source citations — are injected into a structured prompt, sent to Gemini 1.5 Flash or Groq Llama3. The LLM generates a grounded answer, and the UI displays both the answer and source citations like `[DL_Procedure - page 2]`.
>
> Key design decisions: MiniLM for its accuracy-to-size ratio with no API cost. FAISS for zero-setup local vector search. Gemini 1.5 Flash because it's free, has a 1M-token context window, and handles both English and Hindi natively — satisfying the Hindi toggle requirement.
>
> The system is production-aware — handles PDF failures with a fallback knowledge base, uses environment variables for API keys, and is containerized with Docker. The main limitation is fixed-size chunking which can split sentences. This can be improved with semantic chunking or re-ranking. Overall, this demonstrates the full RAG pipeline from data ingestion to grounded generation with citations."

---

*Parivahan RAG Chatbot — SMAI Assignment 3*
