# LLM-Driven Argument Mapping for Health Insurance Policy Documents

Code and data for the paper **"LLM Based Argument Graph Extraction for Reasoning in Health Insurance Policy Documents"** — submitted to ArgMining 2026 / ACL Student Research Workshop.

> We present a pipeline for extracting argument graphs from health insurance policy documents. A dataset of 40 documents (3,671 sentences, 3,007 relations) is constructed from Aetna, CMS, and United Healthcare sources. Six classical baselines and three instruction-tuned LLMs are evaluated under zero-shot, few-shot, and chain-of-thought prompting. TF-IDF+SVM achieves 0.709 Macro F1 on component classification and 0.649 on relation extraction, outperforming the best LLM by 12–21 F1 points while being 18× faster.

---

## Repository Structure

```
LLM-Driven-Argument-Mapping/
├── Data/
│   └── Annotations/
│       ├── Argument_Component/     # Per-document component annotation JSONs
│       └── Argument_Relation/      # Per-document relation annotation JSONs
├── Labelling_Tool/
│   ├── Component_Tool/             # Argument Component Annotation Tool
│   └── Relation_Tool/              # Argument Relation Annotation Tool
├── insurance.ipynb                 # Main experimental pipeline
└── README.md
```

---

## Dataset

The dataset covers health insurance coverage policy documents from three sources:

| Source | Type |
|---|---|
| Aetna Clinical Policy Bulletins | Private insurer |
| CMS National Coverage Determinations | Federal (Medicare/Medicaid) |
| United Healthcare Medical Policies | Private insurer |

| Statistic | Components | Relations |
|---|---|---|
| Documents | 40 | 36 |
| Total instances | 3,671 | 3,007 |
| CLAIM / SUPPORT | 1,092 | 1,526 |
| PREMISE / NONE | 2,017 | 1,046 |
| NON\_ARG / ATTACK | 562 | 435 |
| Train / Test | 2,936 / 735 | 2,405 / 602 |

Inter-annotator agreement: Cohen's κ = 0.82, Krippendorff's α = 0.79.

---

## Models

**Classical baselines:** TF-IDF (unigrams–trigrams) and SBERT embeddings paired with Logistic Regression, Linear SVM, and Random Forest.

**LLMs evaluated:**
- `Qwen2.5-7B-Instruct`
- `Mistral-7B-Instruct-v0.3`
- `Nemotron-Mini-4B-Instruct`

Each LLM is evaluated under **zero-shot**, **few-shot** (2–3 examples per class), and **chain-of-thought** prompting.

---

## Results

### Argument Component Classification

| Model | Prompt | Macro F1 |
|---|---|---|
| TF-IDF + SVM | baseline | **0.709** |
| TF-IDF + LR | baseline | 0.703 |
| Mistral-7B | few-shot | 0.585 |
| Qwen2.5-7B | few-shot | 0.566 |

### Relation Extraction

| Model | Prompt | Macro F1 |
|---|---|---|
| TF-IDF + SVM | baseline | **0.649** |
| TF-IDF + RF | baseline | 0.634 |
| Mistral-7B | CoT | 0.443 |
| Qwen2.5-7B | CoT | 0.440 |

Full results in the paper.

---

## Setup

```bash
git clone https://github.com/GIND123/LLM-Driven-Argument-Mapping
cd LLM-Driven-Argument-Mapping
pip install transformers accelerate bitsandbytes sentence-transformers \
    scikit-learn pandas matplotlib seaborn tqdm networkx
```

A GPU with at least 20GB VRAM is recommended for LLM inference (tested on A40). Classical baselines run on CPU in under 30 seconds.

**Hugging Face login required for LLM models:**
```python
from huggingface_hub import login
login()
```

---

## Running the Pipeline

Open and run `insurance.ipynb` end to end. The notebook is structured as:

1. Load and parse annotation JSONs from `Data/Annotations/`
2. Compute dataset statistics and label distributions
3. Train and evaluate classical baselines (TF-IDF + SBERT × LR/SVM/RF)
4. Run LLM inference under zero-shot / few-shot / CoT prompting
5. Generate result tables and figures
6. Construct argument graphs using NetworkX

Output files are saved to the working directory:
- `component_results.csv`, `relation_results.csv`
- `fig_component_all.pdf`, `fig_relation_all.pdf`
- `fig_prompt_comparison.pdf`, `fig_label_distributions.pdf`
- `fig_comp_heatmap.pdf`, `fig_rel_heatmap.pdf`
- `fig_argument_graph.pdf`

---

## Annotation Tools

See [`Labelling_Tool/README.md`](Labelling_Tool/README.md) for setup and usage instructions for both annotation tools.

---

## Citation

```bibtex
@inproceedings{,
  title     = {LLM Based Argument Graph Extraction for Reasoning in Health Insurance Policy Documents},
  booktitle = {Proceedings of the 11th Workshop on Argument Mining (ArgMining 2026)},
  year      = {2026}
}
```

---

## License

Data and code are released for research use only. Source policy documents are publicly available from Aetna, CMS, and United Healthcare.


