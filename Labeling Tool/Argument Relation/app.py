from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import networkx as nx
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, session, flash
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


ROOT_DIR = Path(__file__).resolve().parents[2]
ARG_COMPONENT_DIR = ROOT_DIR / "Data" / "Annotations" / "Argument_Component"
ARG_RELATION_DIR = ROOT_DIR / "Data" / "Annotations" / "Argument_Relation"

RELATION_OPTIONS = ["SUPPORT", "ATTACK", "NONE"]
VALID_ARGUMENT_LABELS = {"CLAIM", "PREMISE"}

# Diagnostic checks for portability
print(f"--- Argument Relation Tool Startup ---")
print(f"Project Root: {ROOT_DIR}")
print(f"Looking for components in: {ARG_COMPONENT_DIR}")
print(f"Saving relations to: {ARG_RELATION_DIR}")

if not ARG_COMPONENT_DIR.exists():
    print(f"WARNING: Component directory NOT FOUND! Please ensure your folder structure is correct.")
else:
    print(f"SUCCESS: Component directory found.")
print(f"--------------------------------------")


# ---------------------------------------------------------------------------
# Flask App Setup
# ---------------------------------------------------------------------------
app = Flask(__name__)
app.secret_key = "argument-relation-annotation-tool-secret-key"


# ---------------------------------------------------------------------------
# Caching (replaces Streamlit @st.cache_resource / @st.cache_data)
# ---------------------------------------------------------------------------
_embedding_model = None
_embeddings_cache: Dict[tuple, np.ndarray] = {}
_document_cache: Dict[str, Dict] = {}


# ---------------------------------------------------------------------------
# Core Logic — UNCHANGED from the original Streamlit version
# ---------------------------------------------------------------------------

def load_annotation_files() -> List[Path]:
    if not ARG_COMPONENT_DIR.exists():
        return []
    return sorted(ARG_COMPONENT_DIR.glob("*.json"), key=lambda p: p.name.lower())


def _load_document_json(json_path_str: str, mtime_ns: int) -> Dict:
    cache_key = f"{json_path_str}:{mtime_ns}"
    if cache_key in _document_cache:
        return _document_cache[cache_key]
    _ = mtime_ns
    with Path(json_path_str).open("r", encoding="utf-8") as f:
        data = json.load(f)
    _document_cache[cache_key] = data
    return data


def _load_embedding_model() -> SentenceTransformer:
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedding_model


def compute_embeddings(texts: Tuple[str, ...]) -> np.ndarray:
    if texts in _embeddings_cache:
        return _embeddings_cache[texts]
    model = _load_embedding_model()
    embeddings = model.encode(
        list(texts),
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    _embeddings_cache[texts] = embeddings
    return embeddings


def generate_candidate_pairs(sentences: List[Dict]) -> List[Dict]:
    arg_sentences = [s for s in sentences if s.get("label") in VALID_ARGUMENT_LABELS]
    premises = [s for s in arg_sentences if s.get("label") == "PREMISE"]
    claims = [s for s in arg_sentences if s.get("label") == "CLAIM"]

    if not premises or not claims:
        return []

    all_texts = tuple(str(s["text"]) for s in arg_sentences)
    all_embeddings = compute_embeddings(all_texts)
    embedding_by_id = {
        int(sentence["id"]): all_embeddings[idx] for idx, sentence in enumerate(arg_sentences)
    }
    premise_embeddings = np.vstack([embedding_by_id[int(s["id"])] for s in premises])
    claim_embeddings = np.vstack([embedding_by_id[int(s["id"])] for s in claims])
    sim_matrix = cosine_similarity(premise_embeddings, claim_embeddings)

    candidates: List[Dict] = []
    for premise_idx, premise in enumerate(premises):
        row = sim_matrix[premise_idx]
        k = min(3, len(claims))
        top_indices = np.argsort(row)[::-1][:k]
        for claim_idx in top_indices:
            claim = claims[int(claim_idx)]
            candidates.append(
                {
                    "source": int(premise["id"]),
                    "target": int(claim["id"]),
                    "source_text": str(premise["text"]),
                    "target_text": str(claim["text"]),
                    "similarity": float(row[int(claim_idx)]),
                }
            )

    candidates.sort(key=lambda x: (x["source"], -x["similarity"], x["target"]))
    return candidates


def check_graph_constraints(candidates: List[Dict], annotations: Dict[str, str]) -> List[str]:
    warnings: List[str] = []

    outgoing: Dict[int, int] = {}
    for cand in candidates:
        outgoing[cand["source"]] = outgoing.get(cand["source"], 0) + 1
    if any(count > 3 for count in outgoing.values()):
        warnings.append("Constraint warning: A premise has more than 3 candidate claims.")

    graph = nx.DiGraph()
    for cand in candidates:
        key = f"{cand['source']}->{cand['target']}"
        relation_type = annotations.get(key)
        if relation_type in {"SUPPORT", "ATTACK"}:
            graph.add_edge(int(cand["source"]), int(cand["target"]))

    if not nx.is_directed_acyclic_graph(graph):
        warnings.append("Cycle warning: The saved argument graph contains a cycle.")

    return warnings


def save_relations(document: Dict, candidates: List[Dict], annotations: Dict[str, str]) -> Path:
    ARG_RELATION_DIR.mkdir(parents=True, exist_ok=True)

    rows: List[Dict] = []
    for cand in candidates:
        key = f"{cand['source']}->{cand['target']}"
        if key not in annotations:
            continue
        rows.append(
            {
                "source": int(cand["source"]),
                "target": int(cand["target"]),
                "type": annotations[key],
                "similarity": round(float(cand["similarity"]), 4),
            }
        )

    output = {
        "document_id": document.get("document_id", ""),
        "source": document.get("source", ""),
        "relations": rows,
    }

    out_file = ARG_RELATION_DIR / f"{document.get('document_id', 'relations')}.json"
    with out_file.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    return out_file


def _filtered_indices(candidates: List[Dict], annotations: Dict[str, str], mode: str) -> List[int]:
    indices: List[int] = []
    for i, cand in enumerate(candidates):
        key = f"{cand['source']}->{cand['target']}"
        is_annotated = key in annotations
        if mode == "Show Annotated" and not is_annotated:
            continue
        if mode == "Show Unannotated" and is_annotated:
            continue
        indices.append(i)
    return indices


# ---------------------------------------------------------------------------
# In-memory session store (Flask session only stores small data via cookies;
# we keep the heavy objects here, keyed by a simple identifier).
# ---------------------------------------------------------------------------
_app_state: Dict[str, object] = {
    "rel_loaded_doc": None,
    "rel_doc_data": None,
    "rel_candidates": [],
    "rel_annotations": {},
    "rel_filter": "Show All",
    "rel_cursor": 0,
}


# ---------------------------------------------------------------------------
# Flask Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    files = load_annotation_files()
    file_names = [p.name for p in files]

    filter_mode = request.args.get("filter", _app_state.get("rel_filter", "Show All"))
    _app_state["rel_filter"] = filter_mode

    doc_data = _app_state.get("rel_doc_data")
    candidates = _app_state.get("rel_candidates", [])
    annotations = _app_state.get("rel_annotations", {})
    cursor = _app_state.get("rel_cursor", 0)

    # Template context defaults
    ctx = {
        "files": file_names,
        "relation_options": RELATION_OPTIONS,
        "filter_mode": filter_mode,
        "total_sentences": 0,
        "arg_sentences": 0,
        "total_pairs": 0,
        "annotated_count": 0,
        "progress_pct": 0,
        "visible_count": 0,
        "cursor": 0,
        "pair": None,
        "current_saved": RELATION_OPTIONS[0],
    }

    if doc_data is not None:
        total_sentences = len(doc_data.get("sentences", []))
        arg_sents = [s for s in doc_data.get("sentences", []) if s.get("label") in VALID_ARGUMENT_LABELS]
        total_pairs = len(candidates)
        annotated_count = len(annotations)
        progress_pct = (annotated_count / total_pairs * 100) if total_pairs else 0

        visible_indices = _filtered_indices(candidates, annotations, filter_mode)
        visible_count = len(visible_indices)

        # Clamp cursor
        if cursor >= visible_count:
            cursor = max(0, visible_count - 1)
        if cursor < 0:
            cursor = 0
        _app_state["rel_cursor"] = cursor

        pair = None
        current_saved = RELATION_OPTIONS[0]
        if visible_count > 0:
            current_global_idx = visible_indices[cursor]
            pair = candidates[current_global_idx]
            pair_key = f"{pair['source']}->{pair['target']}"
            current_saved = annotations.get(pair_key, RELATION_OPTIONS[0])

        ctx.update({
            "total_sentences": total_sentences,
            "arg_sentences": len(arg_sents),
            "total_pairs": total_pairs,
            "annotated_count": annotated_count,
            "progress_pct": progress_pct,
            "visible_count": visible_count,
            "cursor": cursor,
            "pair": pair,
            "current_saved": current_saved,
        })

    # Mark in session whether doc is loaded (for template conditional)
    session["rel_doc_data"] = doc_data is not None
    session["rel_loaded_doc"] = _app_state.get("rel_loaded_doc", "")

    return render_template("index.html", **ctx)


@app.route("/load", methods=["POST"])
def load_document():
    selected_name = request.form.get("selected_file", "")
    if not selected_name:
        flash("No file selected.", "error")
        return redirect(url_for("index"))

    files = load_annotation_files()
    options = {p.name: p for p in files}

    if selected_name not in options:
        flash(f"File not found: {selected_name}", "error")
        return redirect(url_for("index"))

    path = options[selected_name]
    doc = _load_document_json(str(path), path.stat().st_mtime_ns)
    candidates = generate_candidate_pairs(doc.get("sentences", []))

    _app_state["rel_loaded_doc"] = selected_name
    _app_state["rel_doc_data"] = doc
    _app_state["rel_candidates"] = candidates
    _app_state["rel_annotations"] = {}
    _app_state["rel_filter"] = "Show All"
    _app_state["rel_cursor"] = 0

    flash(f"Loaded: {selected_name}", "success")
    return redirect(url_for("index"))


@app.route("/confirm_relation", methods=["POST"])
def confirm_relation():
    relation_type = request.form.get("relation_type", RELATION_OPTIONS[0])
    candidates = _app_state.get("rel_candidates", [])
    annotations = _app_state.get("rel_annotations", {})
    filter_mode = _app_state.get("rel_filter", "Show All")
    cursor = _app_state.get("rel_cursor", 0)

    visible_indices = _filtered_indices(candidates, annotations, filter_mode)
    if not visible_indices or cursor >= len(visible_indices):
        flash("No pair selected.", "warning")
        return redirect(url_for("index"))

    current_global_idx = visible_indices[cursor]
    pair = candidates[current_global_idx]
    pair_key = f"{pair['source']}->{pair['target']}"

    _app_state["rel_annotations"][pair_key] = relation_type
    flash(f"Saved relation: {pair['source']} → {pair['target']} = {relation_type}", "success")

    # Auto-advance to next pair if possible
    new_visible = _filtered_indices(candidates, _app_state["rel_annotations"], filter_mode)
    if cursor < len(new_visible) - 1:
        _app_state["rel_cursor"] = cursor + 1

    return redirect(url_for("index"))


@app.route("/navigate", methods=["POST"])
def navigate():
    direction = request.form.get("direction", "next")
    cursor = _app_state.get("rel_cursor", 0)

    candidates = _app_state.get("rel_candidates", [])
    annotations = _app_state.get("rel_annotations", {})
    filter_mode = _app_state.get("rel_filter", "Show All")
    visible_indices = _filtered_indices(candidates, annotations, filter_mode)

    if direction == "prev" and cursor > 0:
        _app_state["rel_cursor"] = cursor - 1
    elif direction == "next" and cursor < len(visible_indices) - 1:
        _app_state["rel_cursor"] = cursor + 1

    return redirect(url_for("index"))


@app.route("/save", methods=["POST"])
def save():
    document = _app_state.get("rel_doc_data")
    candidates = _app_state.get("rel_candidates", [])
    annotations = _app_state.get("rel_annotations", {})

    if document is None:
        flash("No document loaded.", "error")
        return redirect(url_for("index"))

    warnings = check_graph_constraints(candidates, annotations)
    out_file = save_relations(document, candidates, annotations)

    for w in warnings:
        flash(w, "warning")
    flash(f"Relations saved successfully. File: {out_file.name}", "success")

    return redirect(url_for("index"))


@app.route("/reset", methods=["POST"])
def reset():
    _app_state["rel_loaded_doc"] = None
    _app_state["rel_doc_data"] = None
    _app_state["rel_candidates"] = []
    _app_state["rel_annotations"] = {}
    _app_state["rel_filter"] = "Show All"
    _app_state["rel_cursor"] = 0
    session.clear()
    flash("Session reset. Select a new document.", "info")
    return redirect(url_for("index"))


if __name__ == "__main__":
    app.run(debug=True, port=5001)
