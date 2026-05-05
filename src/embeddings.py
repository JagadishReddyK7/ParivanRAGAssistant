"""
embeddings.py — Embed chunks with all-MiniLM-L6-v2 and build a FAISS index
Run (from project root): python -m src.embeddings
"""
import json, pickle
from typing import List, Dict, Tuple
import numpy as np
import faiss
from pathlib import Path
from sentence_transformers import SentenceTransformer

# Paths
CHUNKS_FILE  = Path("data/chunks.json")
INDEX_FILE   = Path("data/faiss.index")
META_FILE    = Path("data/metadata.pkl")  # stores chunk metadata aligned with FAISS IDs

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


def build_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)   # inner product = cosine sim (normalized vecs)
    index.add(embeddings)
    return index


def save_index(index: faiss.IndexFlatIP, meta: List[Dict]):
    faiss.write_index(index, str(INDEX_FILE))
    META_FILE.write_bytes(pickle.dumps(meta))
    print(f"✓ FAISS index → {INDEX_FILE}  ({index.ntotal} vectors, dim={index.d})")
    print(f"✓ Metadata    → {META_FILE}")


def load_index() -> Tuple[faiss.IndexFlatIP, List[Dict]]:
    index = faiss.read_index(str(INDEX_FILE))
    meta  = pickle.loads(META_FILE.read_bytes())
    return index, meta


def search(query: str, model: SentenceTransformer,
           index: faiss.IndexFlatIP, meta: List[Dict],
           top_k: int = 5) -> List[Dict]:
    """Return top-k chunks most similar to query."""
    qvec = model.encode([query], convert_to_numpy=True,
                        normalize_embeddings=True).astype("float32")
    scores, ids = index.search(qvec, top_k)
    results = []
    for score, idx in zip(scores[0], ids[0]):
        if idx < 0:
            continue
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
    index      = build_index(embeddings)
    save_index(index, chunks)

    # Quick smoke test
    results = search("how to apply for driving licence", model, index, chunks, top_k=3)
    print("\n── Smoke-test results ──")
    for r in results:
        print(f"  [{r['source']} - page {r['page']}]  score={r['score']:.3f}")
        print(f"  {r['text'][:120]}…\n")
