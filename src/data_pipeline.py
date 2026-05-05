"""
data_pipeline.py — Download & parse Parivahan PDFs into text chunks
Run: python src/data_pipeline.py
"""
import os, json, re
import requests
import fitz  # PyMuPDF
from pathlib import Path
from typing import List, Dict, Optional
from tqdm import tqdm

# ── Configuration ─────────────────────────────────────────────────────────────
# Set to True to use built-in fallback knowledge base if PDF downloads fail.
# Set to False to strictly use local PDFs only (no fallback).
USE_FALLBACK = False

# ── Constants ─────────────────────────────────────────────────────────────────
PDF_DIR = Path("data/pdfs")
CHUNKS_FILE = Path("data/chunks.json")
PDF_DIR.mkdir(parents=True, exist_ok=True)

# Official Parivahan / Sarathi / Vahan PDF sources
# Add more URLs as needed from https://parivahan.gov.in
PDF_SOURCES = [
    {
        "url": "https://sarathi.parivahan.gov.in/SarathiService/pdf/DL_Procedure.pdf",
        "name": "DL_Procedure"
    },
    {
        "url": "https://vahan.parivahan.gov.in/vahanservice/pdf/RC_Procedure.pdf",
        "name": "RC_Procedure"
    },
    # Fallback: use locally placed PDFs in data/pdfs/ if URLs are unavailable
]

# ── Fallback sample text when PDFs can't be fetched (for offline demo) ──
FALLBACK_CHUNKS = [
    {
        "text": (
            "Driving Licence (DL) Application Procedure in India: "
            "Step 1 – Obtain Learner's Licence (LL) by visiting the nearest RTO or applying online "
            "at https://sarathi.parivahan.gov.in. Fill Form 1 (medical fitness) and Form 2. "
            "Pay fee of Rs.200. Appear for LL test after 30 days. "
            "Step 2 – After 30 days of holding LL, apply for permanent DL. Book a driving test slot online. "
            "Bring original LL, Form 4, Form 1A, address proof, age proof, and passport photos. "
            "Step 3 – Appear for driving test at RTO. On passing, DL is issued within 7 days by post."
        ),
        "source": "DL_Procedure_Fallback",
        "page": 1,
        "chunk_id": "DL_Procedure_Fallback_p1_c0"
    },
    {
        "text": (
            "Documents required for Driving Licence: "
            "1. Form 4 (application). 2. Learner's Licence (original). "
            "3. Age proof: Birth Certificate / Class X Marksheet / Passport / Aadhaar. "
            "4. Address proof: Aadhaar / Voter ID / Passport / Electricity Bill. "
            "5. Passport-size photographs (2 copies). 6. Form 1 / Form 1A for medical fitness. "
            "Fee: Rs.200 for LMV licence. Validity: 20 years or until age 50 (whichever is earlier)."
        ),
        "source": "DL_Procedure_Fallback",
        "page": 2,
        "chunk_id": "DL_Procedure_Fallback_p2_c0"
    },
    {
        "text": (
            "Vehicle Registration Certificate (RC) Procedure: "
            "Step 1 – After purchasing a new vehicle, the dealer typically handles RC application. "
            "Step 2 – For re-registration or transfer, visit https://vahan.parivahan.gov.in. "
            "Fill Form 20 for new registration. Submit with insurance certificate, PUC certificate, "
            "Form 21 (sale certificate from dealer), Form 22 (roadworthiness), address proof, ID proof. "
            "Step 3 – Pay road tax and registration fee based on vehicle category. "
            "RC is issued within 7 working days."
        ),
        "source": "RC_Procedure_Fallback",
        "page": 1,
        "chunk_id": "RC_Procedure_Fallback_p1_c0"
    },
    {
        "text": (
            "Documents required for Vehicle Registration (RC): "
            "1. Form 20 (application for registration). "
            "2. Form 21 (sale certificate from the dealer). "
            "3. Form 22 (roadworthiness certificate from manufacturer). "
            "4. Valid insurance certificate. 5. PUC (Pollution Under Control) certificate. "
            "6. Address proof (Aadhaar/Voter ID/Passport). 7. ID proof of the owner. "
            "8. Chassis and engine pencil print. 9. Passport-size photographs. "
            "Fees vary by state and vehicle type. Temporary Registration (TR) valid for 1 month."
        ),
        "source": "RC_Procedure_Fallback",
        "page": 2,
        "chunk_id": "RC_Procedure_Fallback_p2_c0"
    },
    {
        "text": (
            "International Driving Permit (IDP): "
            "Indian citizens can apply for IDP at the issuing RTO. Required documents: valid Indian DL, "
            "passport with valid visa, Form 4A, fee Rs.1000. IDP is valid for 1 year. "
            "Online renewal of Driving Licence: Visit sarathi.parivahan.gov.in → Services → Renewal of DL. "
            "Required: existing DL number, date of birth, Form 1 (medical fitness for >40 years age)."
        ),
        "source": "DL_Procedure_Fallback",
        "page": 3,
        "chunk_id": "DL_Procedure_Fallback_p3_c0"
    },
    {
        "text": (
            "Vehicle RC Transfer Procedure: "
            "When buying a used vehicle, RC transfer is mandatory within 30 days. "
            "Seller submits Form 29 (notice of transfer). Buyer submits Form 30 (intimation of transfer) "
            "along with insurance, PUC, address proof, original RC, NOC if vehicle from different state. "
            "Transfer fee: Rs.300 for two-wheelers, Rs.500 for four-wheelers. "
            "Online: vahan.parivahan.gov.in → Vehicle Related Services → Transfer of Ownership."
        ),
        "source": "RC_Procedure_Fallback",
        "page": 3,
        "chunk_id": "RC_Procedure_Fallback_p3_c0"
    },
]


def download_pdf(url: str, name: str) -> Optional[Path]:
    """Download a PDF and save locally. Returns path or None on failure."""
    dest = PDF_DIR / f"{name}.pdf"
    if dest.exists():
        print(f"  [skip] {name}.pdf already exists")
        return dest
    try:
        r = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        dest.write_bytes(r.content)
        print(f"  [ok]   Downloaded {name}.pdf ({len(r.content)//1024} KB)")
        return dest
    except Exception as e:
        print(f"  [warn] Could not download {name}: {e}")
        return None


def parse_pdf(pdf_path: Path, chunk_size: int = 500) -> List[Dict]:
    """Extract text from PDF, split into overlapping chunks of ~chunk_size words."""
    chunks = []
    doc = fitz.open(str(pdf_path))
    source_name = pdf_path.stem

    for page_num, page in enumerate(doc, start=1):
        text = page.get_text("text").strip()
        text = re.sub(r'\s+', ' ', text)  # normalise whitespace
        if not text:
            continue

        # Split into word-level chunks with 50-word overlap
        words = text.split()
        step = chunk_size - 50  # overlap
        for i, start in enumerate(range(0, len(words), step)):
            chunk_text = " ".join(words[start: start + chunk_size])
            if len(chunk_text) < 80:  # skip tiny fragments
                continue
            chunks.append({
                "text": chunk_text,
                "source": source_name,
                "page": page_num,
                "chunk_id": f"{source_name}_p{page_num}_c{i}"
            })

    doc.close()
    return chunks


def build_chunks() -> List[Dict]:
    """Download PDFs, parse them, and save chunks to JSON. Falls back to sample data."""
    all_chunks = []
    pdf_found = False

    print("=== Downloading Parivahan PDFs ===")
    for src in tqdm(PDF_SOURCES, desc="PDFs"):
        path = download_pdf(src["url"], src["name"])
        if path:
            chunks = parse_pdf(path)
            all_chunks.extend(chunks)
            pdf_found = True
            print(f"  → {len(chunks)} chunks from {src['name']}")

    # Also parse any manually placed PDFs in data/pdfs/
    for pdf_file in PDF_DIR.glob("*.pdf"):
        name = pdf_file.stem
        if not any(s["name"] == name for s in PDF_SOURCES):
            chunks = parse_pdf(pdf_file)
            all_chunks.extend(chunks)
            pdf_found = True
            print(f"  → {len(chunks)} chunks from {name} (manual)")

    if not pdf_found:
        if USE_FALLBACK:
            print("\n[info] No PDFs fetched — using built-in fallback knowledge base.")
            all_chunks = FALLBACK_CHUNKS
        else:
            print("\n[warn] No PDFs fetched and USE_FALLBACK is False. Index will be empty!")
            all_chunks = []

    CHUNKS_FILE.parent.mkdir(parents=True, exist_ok=True)
    CHUNKS_FILE.write_text(json.dumps(all_chunks, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[ok] Saved {len(all_chunks)} chunks -> {CHUNKS_FILE}")
    return all_chunks


if __name__ == "__main__":
    build_chunks()
