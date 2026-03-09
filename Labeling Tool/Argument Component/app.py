from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Dict, List

import pandas as pd
import pdfplumber
import streamlit as st

# Prevent spaCy/thinc from importing torch, which may be broken in some local setups.
os.environ.setdefault("THINC_IGNORE_TORCH", "1")
import spacy


ROOT_DIR = Path(__file__).resolve().parents[2]
ARG_COMPONENT_OUT_DIR = ROOT_DIR / "Data" / "Annotations" / "Argument_Component"

LABEL_OPTIONS = ["", "CLAIM", "PREMISE", "NON_ARGUMENTATIVE"]
LABEL_COLORS = {
    "CLAIM": "#f8d7da",
    "PREMISE": "#d1f2d9",
    "NON_ARGUMENTATIVE": "#e2e3e5",
    "": "#ffffff",
}


@st.cache_resource
def load_spacy_model():
    # Keep required model, but disable heavy components and use rule-based sentence splitting.
    nlp = spacy.load(
        "en_core_web_sm",
        disable=["tok2vec", "tagger", "parser", "attribute_ruler", "lemmatizer", "ner"],
    )
    if "sentencizer" not in nlp.pipe_names:
        nlp.add_pipe("sentencizer")
    return nlp


def normalize_doc_id(pdf_name: str) -> str:
    stem = Path(pdf_name).stem
    return re.sub(r"\s+", "_", stem).strip("_")


def list_pdf_files(pdf_dir: Path) -> List[Path]:
    if not pdf_dir.exists():
        return []
    return sorted(pdf_dir.glob("*.pdf"), key=lambda p: p.name.lower())


def get_raw_pdf_dir() -> Path:
    # Support both folder spellings that may exist in this project.
    candidates = [
        ROOT_DIR / "Data" / "Raw PDF" / "aetna",
        ROOT_DIR / "Data" / "Raw PDf" / "aetna",
    ]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


@st.cache_data(show_spinner=False)
def extract_text_from_pdf_cached(pdf_path_str: str, mtime_ns: int) -> str:
    _ = mtime_ns  # invalidate cache when file changes
    text_chunks: List[str] = []
    with pdfplumber.open(pdf_path_str) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text() or ""
            if page_text.strip():
                text_chunks.append(page_text)
    return "\n".join(text_chunks)


def extract_text_from_pdf(pdf_path: Path) -> str:
    return extract_text_from_pdf_cached(str(pdf_path), pdf_path.stat().st_mtime_ns)


@st.cache_data(show_spinner=False)
def split_into_sentences_cached(text: str) -> List[Dict[str, object]]:
    nlp = load_spacy_model()
    doc = nlp(text)
    sentences: List[Dict[str, object]] = []
    idx = 1
    for sent in doc.sents:
        sent_text = sent.text.strip()
        if not sent_text:
            continue
        sentences.append({"id": idx, "text": sent_text})
        idx += 1
    return sentences


def split_into_sentences(text: str) -> List[Dict[str, object]]:
    return split_into_sentences_cached(text)


def initialize_state():
    st.session_state.setdefault("loaded", False)
    st.session_state.setdefault("current_pdf_name", None)
    st.session_state.setdefault("document_id", None)
    st.session_state.setdefault("sentences", [])
    st.session_state.setdefault("labels", {})


def load_document(pdf_path: Path):
    text = extract_text_from_pdf(pdf_path)
    sentences = split_into_sentences(text)
    labels = {item["id"]: "" for item in sentences}

    st.session_state.loaded = True
    st.session_state.current_pdf_name = pdf_path.name
    st.session_state.document_id = normalize_doc_id(pdf_path.name)
    st.session_state.sentences = sentences
    st.session_state.labels = labels

    for item in sentences:
        sid = item["id"]
        st.session_state[f"label_selector_{sid}"] = ""


def set_label(sentence_id: int, label: str):
    st.session_state.labels[sentence_id] = label


def save_annotations():
    ARG_COMPONENT_OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(st.session_state.sentences)
    df["label"] = df["id"].map(lambda sid: st.session_state.labels.get(int(sid), ""))
    df = df[df["label"].astype(str).str.strip() != ""]
    sentences_payload = df.to_dict(orient="records")

    payload = {
        "document_id": st.session_state.document_id,
        "source": "Aetna Clinical Policy Bulletin",
        "sentences": sentences_payload,
    }

    output_file = ARG_COMPONENT_OUT_DIR / f"{st.session_state.document_id}.json"
    with output_file.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    return output_file


def render_sentence_card(sentence: Dict[str, object]):
    sid = int(sentence["id"])
    text = str(sentence["text"])
    label = st.session_state.labels.get(sid, "")
    bg_color = LABEL_COLORS.get(label, "#ffffff")

    with st.container(border=True):
        st.markdown(f"**Sentence {sid}**")

        st.markdown(
            f"""
            <div style="background-color:{bg_color}; color:#111111; padding:0.75rem; border-radius:0.4rem; border:1px solid #d0d7de;">
              {text}
            </div>
            """,
            unsafe_allow_html=True,
        )

        if f"label_selector_{sid}" not in st.session_state:
            st.session_state[f"label_selector_{sid}"] = label

        with st.form(key=f"form_{sid}", clear_on_submit=False):
            select_col, action_col = st.columns([0.7, 0.3])
            with select_col:
                st.selectbox(
                    "Label",
                    options=LABEL_OPTIONS,
                    key=f"label_selector_{sid}",
                )
            with action_col:
                submitted = st.form_submit_button("Confirm Label", use_container_width=True)
            if submitted:
                chosen = st.session_state.get(f"label_selector_{sid}", "")
                set_label(sid, chosen)
                st.rerun()


def main():
    st.set_page_config(page_title="Argument Component Annotation Tool", layout="wide")
    initialize_state()

    st.title("Argument Component Annotation Tool")

    raw_pdf_dir = get_raw_pdf_dir()
    pdf_files = list_pdf_files(raw_pdf_dir)
    if not pdf_files:
        st.error(f"No PDF files found in: {raw_pdf_dir}")
        return

    pdf_options = {p.name: p for p in pdf_files}
    selected_pdf_name = st.selectbox("Select a PDF", options=list(pdf_options.keys()))

    if st.button("Load Document", type="primary"):
        with st.spinner("Loading, extracting text, and splitting sentences..."):
            load_document(pdf_options[selected_pdf_name])
        st.success(f"Loaded: {selected_pdf_name}")

    if not st.session_state.loaded:
        st.info("Load a document to begin annotation.")
        return

    stats_df = pd.DataFrame(st.session_state.sentences)
    if not stats_df.empty:
        stats_df["label"] = stats_df["id"].map(lambda sid: st.session_state.labels.get(int(sid), ""))
    total_sentences = int(stats_df.shape[0])
    labeled_sentences = int((stats_df["label"] != "").sum()) if total_sentences else 0
    remaining_sentences = total_sentences - labeled_sentences
    progress = (labeled_sentences / total_sentences) if total_sentences else 0.0

    st.subheader(f"Document: {st.session_state.current_pdf_name}")

    stat_col1, stat_col2, stat_col3 = st.columns(3)
    stat_col1.metric("Total sentences", total_sentences)
    stat_col2.metric("Labeled sentences", labeled_sentences)
    stat_col3.metric("Remaining sentences", remaining_sentences)

    st.progress(progress, text=f"Progress: {int(progress * 100)}% labeled")

    st.caption("Select a label and click Confirm Label to commit the annotation.")

    filter_option = st.radio(
        "Filter",
        options=["Show All", "Show Unlabeled", "Show CLAIM", "Show PREMISE", "Show NON_ARGUMENTATIVE"],
        horizontal=True,
    )

    if filter_option == "Show All":
        visible_sentences = st.session_state.sentences
    elif filter_option == "Show Unlabeled":
        visible_sentences = [
            s for s in st.session_state.sentences if not st.session_state.labels.get(int(s["id"]), "")
        ]
    else:
        target = filter_option.replace("Show ", "")
        visible_sentences = [
            s for s in st.session_state.sentences if st.session_state.labels.get(int(s["id"]), "") == target
        ]

    st.markdown("### Sentences")
    try:
        sentence_holder = st.container(height=520)
    except TypeError:
        sentence_holder = st.container()

    with sentence_holder:
        if not visible_sentences:
            st.info("No sentences match the current filter.")
        for sent in visible_sentences:
            render_sentence_card(sent)

    if st.button("Save Annotations", type="primary"):
        output_file = save_annotations()
        st.success(f"Annotations saved successfully. File: {output_file.name}")


if __name__ == "__main__":
    main()
