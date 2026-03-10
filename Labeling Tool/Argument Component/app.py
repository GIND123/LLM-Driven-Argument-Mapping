from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Dict, List

import pandas as pd
import pdfplumber
from flask import Flask, jsonify, render_template, request

# Prevent spaCy/thinc from importing torch, which may be broken in some local setups.
os.environ.setdefault("THINC_IGNORE_TORCH", "1")
import spacy

app = Flask(__name__)

ROOT_DIR = Path(__file__).resolve().parents[2]
ARG_COMPONENT_OUT_DIR = ROOT_DIR / "Data" / "Annotations" / "Argument_Component"

LABEL_OPTIONS = ["", "CLAIM", "PREMISE", "NON_ARGUMENTATIVE"]
LABEL_COLORS = {
    "CLAIM": "#f8d7da",
    "PREMISE": "#d1f2d9",
    "NON_ARGUMENTATIVE": "#e2e3e5",
    "": "#ffffff",
}

# --- Core Logic from original app.py ---

_spacy_model = None

def load_spacy_model():
    global _spacy_model
    if _spacy_model is None:
        _spacy_model = spacy.load(
            "en_core_web_sm",
            disable=["tok2vec", "tagger", "parser", "attribute_ruler", "lemmatizer", "ner"],
        )
        if "sentencizer" not in _spacy_model.pipe_names:
            _spacy_model.add_pipe("sentencizer")
    return _spacy_model

def normalize_doc_id(pdf_name: str) -> str:
    stem = Path(pdf_name).stem
    return re.sub(r"\s+", "_", stem).strip("_")

def list_pdf_files(pdf_dir: Path) -> List[Path]:
    if not pdf_dir.exists():
        return []
    return sorted(pdf_dir.glob("*.pdf"), key=lambda p: p.name.lower())

def get_raw_pdf_dir() -> Path:
    candidates = [
        ROOT_DIR / "Data" / "Raw PDF" / "aetna",
        ROOT_DIR / "Data" / "Raw PDf" / "aetna",
    ]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]

_NOISE_PATTERNS: List[re.Pattern] = [
    re.compile(r"Explore\s+our\s+developer[\-\s]*friendly\s+HTML\s+to\s+PDF\s+API", re.IGNORECASE),
    re.compile(r"Printed\s+using\s+PDFCrowd\s+HTML\s+to\s+PDF", re.IGNORECASE),
    re.compile(r"PDFCrowd\s+HTML\s+to\s+PDF", re.IGNORECASE),
]

_JUNK_LINE_RES: List[re.Pattern] = [
    re.compile(r"^\s*Table\s+Of\s+Contents\s*$", re.IGNORECASE),
    re.compile(r"^\s*Page\s+\d+\s+of\s+\d+\s*$", re.IGNORECASE),
    re.compile(r"^\s*--+>\s*$"),
    re.compile(r"^\s*Policy\s*$", re.IGNORECASE),
    re.compile(r"^\s*References\s*$", re.IGNORECASE),
    re.compile(r"^\s*Background\s*$", re.IGNORECASE),
]

_HTML_TAG_RE = re.compile(r"<[^>]+>")
_HTML_ENTITY_RE = re.compile(r"&(?:#\d+|#x[0-9a-fA-F]+|[a-zA-Z]+);")

def _clean_extracted_text(raw_text: str) -> str:
    text = raw_text
    text = _HTML_TAG_RE.sub(" ", text)
    text = _HTML_ENTITY_RE.sub(" ", text)
    for pat in _NOISE_PATTERNS:
        text = pat.sub(" ", text)
    kept_lines: List[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if any(pat.match(stripped) for pat in _JUNK_LINE_RES):
            continue
        kept_lines.append(stripped)
    text = " ".join(kept_lines)
    text = re.sub(r"\s{2,}", " ", text).strip()
    return text

def extract_text_from_pdf(pdf_path: Path) -> str:
    text_chunks: List[str] = []
    with pdfplumber.open(str(pdf_path)) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text() or ""
            if page_text.strip():
                text_chunks.append(page_text)
    raw = "\n".join(text_chunks)
    return _clean_extracted_text(raw)

_MIN_SENT_LEN = 10
_MIN_ALPHA = 3
_MERGE_THRESHOLD = 40
_ROMAN_SECTION_RE = re.compile(r"(?<=\s)(?=(?:X{0,3})(?:IX|IV|V?I{0,3})\.\s+[A-Z][a-z])")

def _is_meaningful(text: str) -> bool:
    return len(text) >= _MIN_SENT_LEN and sum(1 for c in text if c.isalpha()) >= _MIN_ALPHA

def split_into_sentences(text: str) -> List[Dict[str, object]]:
    nlp = load_spacy_model()
    sections = [s.strip() for s in _ROMAN_SECTION_RE.split(text) if s.strip()]
    raw_frags: List[str] = []
    for section in sections:
        doc = nlp(section)
        for sent in doc.sents:
            frag = sent.text.strip()
            if frag:
                raw_frags.append(frag)
    merged: List[str] = []
    for frag in raw_frags:
        if merged and len(frag) < _MERGE_THRESHOLD:
            merged[-1] = merged[-1].rstrip() + " " + frag
        else:
            merged.append(frag)
    sentences: List[Dict[str, object]] = []
    idx = 1
    for frag in merged:
        frag = frag.strip()
        if not _is_meaningful(frag):
            continue
        sentences.append({"id": idx, "text": frag})
        idx += 1
    return sentences

# --- Flask Routes ---

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/pdfs", methods=["GET"])
def get_pdfs():
    raw_pdf_dir = get_raw_pdf_dir()
    pdf_files = list_pdf_files(raw_pdf_dir)
    return jsonify({"pdfs": [p.name for p in pdf_files]})

@app.route("/api/load", methods=["POST"])
def load_pdf():
    data = request.json
    pdf_name = data.get("pdf_name")
    if not pdf_name:
        return jsonify({"error": "No pdf_name provided"}), 400
    
    raw_pdf_dir = get_raw_pdf_dir()
    pdf_path = raw_pdf_dir / pdf_name
    
    if not pdf_path.exists():
        return jsonify({"error": "PDF not found"}), 404
        
    text = extract_text_from_pdf(pdf_path)
    sentences = split_into_sentences(text)
    doc_id = normalize_doc_id(pdf_name)
    
    return jsonify({
        "document_id": doc_id,
        "sentences": sentences
    })

@app.route("/api/save", methods=["POST"])
def save():
    data = request.json
    doc_id = data.get("document_id")
    annotations = data.get("sentences", [])
    
    if not doc_id:
        return jsonify({"error": "Missing document_id"}), 400
        
    ARG_COMPONENT_OUT_DIR.mkdir(parents=True, exist_ok=True)
    
    filtered_sentences = [
        {"id": s.get("id"), "text": s.get("text"), "label": s.get("label")} 
        for s in annotations
        if s.get("label", "").strip() != ""
    ]
    
    payload = {
        "document_id": doc_id,
        "source": "Aetna Clinical Policy Bulletin",
        "sentences": filtered_sentences,
    }

    output_file = ARG_COMPONENT_OUT_DIR / f"{doc_id}.json"
    with output_file.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        
    return jsonify({"message": f"Annotations saved to {output_file.name}"})

if __name__ == "__main__":
    app.run(debug=True, port=8501)
