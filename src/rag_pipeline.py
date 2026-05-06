"""
rag_pipeline.py — Retrieve relevant chunks, build prompt, call LLM (Gemini or Groq)
"""
import os
from typing import List, Dict, Tuple
from dotenv import load_dotenv

load_dotenv(override=True)

TOP_K = 5

# Read at call time so .env loaded by app takes effect
def _provider() -> str:
    return os.getenv("LLM_PROVIDER", "gemini").lower()

def _gemini_key() -> str:
    return os.getenv("GEMINI_API_KEY", "")

def _groq_key() -> str:
    return os.getenv("GROQ_API_KEY", "")

# ── System prompt template ────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are a knowledgeable assistant specialising in Indian motor vehicle services
(Driving Licence, Vehicle RC, Sarathi, Vahan). Answer ONLY from the provided context.
If the context doesn't contain the answer, say "I don't have enough information."

Rules:
- Be concise and step-by-step where applicable.
- Do NOT hallucinate fees, URLs, or form numbers not present in the context.
{hindi_instruction}
"""

HINDI_INSTRUCTION = "Respond in Hindi (Devanagari script). Keep citations in English."


def build_prompt(query: str, chunks: List[Dict], hindi: bool = False) -> Tuple[str, str]:
    """Return (system_prompt, user_message) ready for LLM."""
    hindi_instr = HINDI_INSTRUCTION if hindi else ""
    system = SYSTEM_PROMPT.format(hindi_instruction=hindi_instr).strip()

    context_blocks = []
    for c in chunks:
        citation = f"[{c['source']} - page {c['page']}]"
        context_blocks.append(f"{citation}\n{c['text']}")

    context_str = "\n\n---\n\n".join(context_blocks)
    user_msg = f"Context:\n{context_str}\n\nQuestion: {query}"
    return system, user_msg


# ── Gemini 2.0 Flash (direct REST, v1beta endpoint) ──────────────────────────
def call_gemini(system: str, user_msg: str) -> str:
    import requests, time
    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"gemini-2.0-flash:generateContent?key={_gemini_key()}"
    )
    full_content = f"{system}\n\n{user_msg}"
    payload = {
        "contents": [{"role": "user", "parts": [{"text": full_content}]}],
        "generationConfig": {"temperature": 0.2, "maxOutputTokens": 1024}
    }
    for attempt in range(3):               # retry up to 3 times on 429
        r = requests.post(url, json=payload, timeout=30)
        if r.status_code == 429:
            wait = 15 * (attempt + 1)      # wait 15s, 30s, 45s
            time.sleep(wait)
            continue
        r.raise_for_status()
        return r.json()["candidates"][0]["content"]["parts"][0]["text"]
    r.raise_for_status()                   # raise after 3 failed attempts


# ── Groq Llama3 ───────────────────────────────────────────────────────────────
def call_groq(system: str, user_msg: str) -> str:
    import requests
    headers = {
        "Authorization": f"Bearer {_groq_key()}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "llama-3.3-70b-versatile",
        "messages": [
            {"role": "system", "content": system},
            {"role": "user",   "content": user_msg}
        ],
        "temperature": 0.2,
        "max_tokens": 1024
    }
    r = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        json=payload, headers=headers, timeout=30
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


# ── Public interface ──────────────────────────────────────────────────────────
def answer_query(query: str, chunks: List[Dict], hindi: bool = False) -> Tuple[str, List[Dict]]:
    """
    Given a query and retrieved chunks, call the configured LLM.
    Returns (answer_text, source_chunks).
    """
    system, user_msg = build_prompt(query, chunks, hindi)

    try:
        if _provider() == "groq":
            answer = call_groq(system, user_msg)
        else:
            answer = call_gemini(system, user_msg)
    except Exception as e:
        answer = (
            f"⚠️ LLM error ({_provider()}): {e}\n\nRetrieved context:\n" +
            "\n".join(f"[{c['source']} p{c['page']}] {c['text'][:200]}" for c in chunks)
        )

    return answer, chunks


def format_citations(chunks: List[Dict]) -> str:
    """Return a deduplicated citation block for display."""
    seen = set()
    lines = []
    for c in chunks:
        key = (c["source"], c["page"])
        if key not in seen:
            seen.add(key)
            lines.append(f"📄 `{c['source']}` — page {c['page']}")
    return "\n".join(lines)
