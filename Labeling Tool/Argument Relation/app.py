from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import networkx as nx
import numpy as np
import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


ROOT_DIR = Path(__file__).resolve().parents[2]
ARG_COMPONENT_DIR = ROOT_DIR / "Data" / "Annotations" / "Argument_Component"
ARG_RELATION_DIR = ROOT_DIR / "Data" / "Annotations" / "Argument_Relation"

RELATION_OPTIONS = ["SUPPORT", "ATTACK", "NONE"]
VALID_ARGUMENT_LABELS = {"CLAIM", "PREMISE"}


def load_annotation_files() -> List[Path]:
    if not ARG_COMPONENT_DIR.exists():
        return []
    return sorted(ARG_COMPONENT_DIR.glob("*.json"), key=lambda p: p.name.lower())


@st.cache_data(show_spinner=False)
def load_document(json_path_str: str, mtime_ns: int) -> Dict:
    _ = mtime_ns
    with Path(json_path_str).open("r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_resource
def _load_embedding_model() -> SentenceTransformer:
    return SentenceTransformer("all-MiniLM-L6-v2")


@st.cache_data(show_spinner=False)
def compute_embeddings(texts: Tuple[str, ...]) -> np.ndarray:
    model = _load_embedding_model()
    embeddings = model.encode(
        list(texts),
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
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


def _initialize_state():
    st.session_state.setdefault("rel_loaded_doc", None)
    st.session_state.setdefault("rel_doc_data", None)
    st.session_state.setdefault("rel_candidates", [])
    st.session_state.setdefault("rel_annotations", {})
    st.session_state.setdefault("rel_filter", "Show All")
    st.session_state.setdefault("rel_cursor", 0)


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


def relation_annotation_interface():
    candidates = st.session_state.rel_candidates
    annotations = st.session_state.rel_annotations
    document = st.session_state.rel_doc_data

    total_sentences = len(document.get("sentences", []))
    arg_sentences = [s for s in document.get("sentences", []) if s.get("label") in VALID_ARGUMENT_LABELS]
    total_pairs = len(candidates)
    annotated_count = len(annotations)

    st.subheader(f"Document: {st.session_state.rel_loaded_doc}")
    c1, c2, c3 = st.columns(3)
    c1.metric("Total sentences", total_sentences)
    c2.metric("Argument sentences", len(arg_sentences))
    c3.metric("Candidate relation pairs", total_pairs)
    st.progress(annotated_count / total_pairs if total_pairs else 0, text=f"Progress: {annotated_count} / {total_pairs}")

    filter_mode = st.radio(
        "Filter",
        options=["Show All", "Show Annotated", "Show Unannotated"],
        horizontal=True,
        key="rel_filter_radio",
    )
    st.session_state.rel_filter = filter_mode

    visible_indices = _filtered_indices(candidates, annotations, filter_mode)
    if not visible_indices:
        st.info("No candidate pairs in this filter.")
        return

    if st.session_state.rel_cursor >= len(visible_indices):
        st.session_state.rel_cursor = len(visible_indices) - 1
    if st.session_state.rel_cursor < 0:
        st.session_state.rel_cursor = 0

    current_global_idx = visible_indices[st.session_state.rel_cursor]
    pair = candidates[current_global_idx]
    pair_key = f"{pair['source']}->{pair['target']}"
    current_saved = annotations.get(pair_key, RELATION_OPTIONS[0])

    st.markdown("### Candidate Pair")
    st.markdown(
        f"""
        <div style="background-color:#d1f2d9; color:#111; padding:0.75rem; border-radius:0.4rem; border:1px solid #b5dfbe;">
          <b>PREMISE (ID {pair['source']})</b><br>{pair['source_text']}
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        f"""
        <div style="background-color:#f8d7da; color:#111; padding:0.75rem; border-radius:0.4rem; border:1px solid #eab5bc;">
          <b>CLAIM (ID {pair['target']})</b><br>{pair['target_text']}
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption(f"Cosine similarity: {pair['similarity']:.4f}")

    with st.form("relation_form", clear_on_submit=False):
        relation = st.selectbox(
            "Relation type",
            options=RELATION_OPTIONS,
            index=RELATION_OPTIONS.index(current_saved) if current_saved in RELATION_OPTIONS else 0,
        )
        confirmed = st.form_submit_button("Confirm Relation", use_container_width=True)
        if confirmed:
            st.session_state.rel_annotations[pair_key] = relation
            st.success(f"Saved relation: {pair['source']} -> {pair['target']} = {relation}")
            st.rerun()

    nav_prev, nav_next = st.columns(2)
    with nav_prev:
        if st.button("Previous Pair", use_container_width=True, disabled=st.session_state.rel_cursor <= 0):
            st.session_state.rel_cursor -= 1
            st.rerun()
    with nav_next:
        if st.button(
            "Next Pair",
            use_container_width=True,
            disabled=st.session_state.rel_cursor >= len(visible_indices) - 1,
        ):
            st.session_state.rel_cursor += 1
            st.rerun()

    if st.button("Save Relations", type="primary", use_container_width=True):
        warnings = check_graph_constraints(candidates, annotations)
        out_file = save_relations(document, candidates, annotations)
        for w in warnings:
            st.warning(w)
        st.success(f"Relations saved successfully. File: {out_file.name}")


def main():
    st.set_page_config(page_title="Argument Relation Annotation Tool", layout="wide")
    _initialize_state()
    st.title("Argument Relation Annotation Tool")

    files = load_annotation_files()
    if not files:
        st.error(f"No annotation JSON files found in: {ARG_COMPONENT_DIR}")
        return

    options = {p.name: p for p in files}
    selected_name = st.selectbox("Select a document", list(options.keys()))

    if st.button("Load Document", type="primary"):
        with st.spinner("Loading document and generating candidate pairs..."):
            doc = load_document(str(options[selected_name]), options[selected_name].stat().st_mtime_ns)
            candidates = generate_candidate_pairs(doc.get("sentences", []))
        st.session_state.rel_loaded_doc = selected_name
        st.session_state.rel_doc_data = doc
        st.session_state.rel_candidates = candidates
        st.session_state.rel_annotations = {}
        st.session_state.rel_filter = "Show All"
        st.session_state.rel_cursor = 0
        st.success(f"Loaded: {selected_name}")

    if st.session_state.rel_doc_data is None:
        st.info("Load a document to begin relation annotation.")
        return

    relation_annotation_interface()


if __name__ == "__main__":
    main()
