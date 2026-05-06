"""
embeddings.py — Embed chunks with all-MiniLM-L6-v2 and build a FAISS index
Run (from project root): python -m src.embeddings
"""
import json, pickle
from typing import List, Dict, Tuple
import numpy as np
import faiss
import re
from pathlib import Path
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

# Paths
CHUNKS_FILE  = Path("data/chunks.json")
INDEX_FILE   = Path("data/faiss.index")
META_FILE    = Path("data/metadata.pkl")  # stores chunk metadata aligned with FAISS IDs
BM25_FILE    = Path("data/bm25.pkl")      # stores BM25 index

MODEL_NAME   = "sentence-transformers/all-MiniLM-L6-v2"
BATCH_SIZE   = 64


def load_model() -> SentenceTransformer:
    print(f"Loading embedding model: {MODEL_NAME}")
    return SentenceTransformer(MODEL_NAME, device="cpu")


def embed_chunks(model: SentenceTransformer, chunks: List[Dict]) -> np.ndarray:
    texts = [c["text"] for c in chunks]
    print(f"Embedding {len(texts)} chunks in batches of {BATCH_SIZE}…")
    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True  # cosine sim via inner product
    )
    return embeddings.astype("float32")


def tokenize(text: str) -> List[str]:
    """Simple tokenizer for BM25: lowercase and split on non-alphanumeric."""
    return [word for word in re.split(r'\W+', text.lower()) if word]


def build_index(embeddings: np.ndarray, chunks: List[Dict]) -> Tuple[faiss.IndexFlatIP, BM25Okapi]:
    # 1. Build FAISS
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)   # inner product = cosine sim (normalized vecs)
    index.add(embeddings)
    
    # 2. Build BM25
    tokenized_corpus = [tokenize(c["text"]) for c in chunks]
    bm25 = BM25Okapi(tokenized_corpus)
    
    return index, bm25


def save_index(index: faiss.IndexFlatIP, bm25: BM25Okapi, meta: List[Dict]):
    faiss.write_index(index, str(INDEX_FILE))
    META_FILE.write_bytes(pickle.dumps(meta))
    BM25_FILE.write_bytes(pickle.dumps(bm25))
    print(f"[ok] FAISS index -> {INDEX_FILE}  ({index.ntotal} vectors, dim={index.d})")
    print(f"[ok] BM25 index  -> {BM25_FILE}")
    print(f"[ok] Metadata    -> {META_FILE}")


def load_index() -> Tuple[faiss.IndexFlatIP, BM25Okapi, List[Dict]]:
    index = faiss.read_index(str(INDEX_FILE))
    meta  = pickle.loads(META_FILE.read_bytes())
    bm25  = pickle.loads(BM25_FILE.read_bytes())
    return index, bm25, meta


def search(query: str, model: SentenceTransformer,
           index: faiss.IndexFlatIP, bm25: BM25Okapi, meta: List[Dict],
           top_k: int = 5) -> List[Dict]:
    """Hybrid search (FAISS + BM25) with Reciprocal Rank Fusion (RRF)."""
    
    # We fetch a larger pool from each retriever to merge
    pool_size = max(top_k * 2, 10)
    
    # 1. FAISS Search
    qvec = model.encode([query], convert_to_numpy=True,
                        normalize_embeddings=True).astype("float32")
    faiss_scores, faiss_ids = index.search(qvec, pool_size)
    faiss_ranks = {int(idx): rank for rank, idx in enumerate(faiss_ids[0]) if idx >= 0}
    
    # 2. BM25 Search
    tokenized_query = tokenize(query)
    bm25_scores = bm25.get_scores(tokenized_query)
    # Get top pool_size indices
    bm25_ids = np.argsort(bm25_scores)[::-1][:pool_size]
    # Filter out zero scores to avoid ranking irrelevant docs
    bm25_ids = [idx for idx in bm25_ids if bm25_scores[idx] > 0]
    bm25_ranks = {int(idx): rank for rank, idx in enumerate(bm25_ids)}
    
    # 3. Reciprocal Rank Fusion (RRF)
    # RRF_Score = 1 / (k + rank)  -- k=60 is standard
    k = 60
    all_indices = set(faiss_ranks.keys()).union(set(bm25_ranks.keys()))
    
    rrf_scores = []
    for idx in all_indices:
        score = 0.0
        if idx in faiss_ranks:
            score += 1.0 / (k + faiss_ranks[idx])
        if idx in bm25_ranks:
            score += 1.0 / (k + bm25_ranks[idx])
        rrf_scores.append((score, idx))
        
    # Sort by descending RRF score
    rrf_scores.sort(key=lambda x: x[0], reverse=True)
    
    # 4. Build results
    results = []
    for score, idx in rrf_scores[:top_k]:
        entry = meta[idx].copy()
        entry["score"] = float(score)
        results.append(entry)
        
    return results


if __name__ == "__main__":
    import sys as _sys
    # Ensure project root (parent of src/) is on path
    _root = str(Path(__file__).resolve().parent.parent)
    if _root not in _sys.path:
        _sys.path.insert(0, _root)

    if not CHUNKS_FILE.exists():
        print("[info] chunks.json not found — running data pipeline first…")
        from src.data_pipeline import build_chunks
        chunks = build_chunks()
    else:
        chunks = json.loads(CHUNKS_FILE.read_text(encoding="utf-8"))

    model      = load_model()
    embeddings = embed_chunks(model, chunks)
    index, bm25 = build_index(embeddings, chunks)
    save_index(index, bm25, chunks)

    # Quick smoke test
    results = search("form 4", model, index, bm25, chunks, top_k=3)
    print("\n── Smoke-test results ──")
    for r in results:
        print(f"  [{r['source']} - page {r['page']}]  score={r['score']:.3f}")
        print(f"  {r['text'][:120]}…\n")
