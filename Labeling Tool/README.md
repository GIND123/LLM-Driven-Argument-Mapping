# Annotation Tools

Two browser-based tools for annotating argument components and relations in policy documents. Both tools export structured JSON compatible with the main pipeline.

---

## Argument Component Annotation Tool

Labels each sentence in a policy document as `CLAIM`, `PREMISE`, or `NON_ARGUMENTATIVE`.

**Workflow:**
1. Select a dataset (Aetna / CMS / United Healthcare) and load a policy PDF
2. The tool extracts and sentence-segments the text using spaCy
3. Each sentence is presented as a numbered card
4. Assign a label via the dropdown (`CLAIM`, `PREMISE`, `NON_ARGUMENTATIVE`)
5. Progress is tracked (total / labelled / remaining)
6. Export the completed annotation as a JSON file

**Output format (`Argument_Component/<doc_id>.json`):**
```json
{
  "document_id": "Alzheimer_Disease_Tests",
  "sentences": [
    {
      "sentence_id": 1,
      "text": "Aetna considers Brain PET scans medically necessary...",
      "label": "CLAIM"
    }
  ]
}
```

---

## Argument Relation Annotation Tool

Labels pairwise relations between argument components as `SUPPORT`, `ATTACK`, or `NONE`.

**Workflow:**
1. Load a completed component annotation JSON
2. The tool automatically generates candidate premise–claim pairs using cosine similarity over SBERT (`all-MiniLM-L6-v2`) embeddings — top-3 most similar claims per premise are retained
3. Each candidate pair is shown with its cosine similarity score
4. Assign a relation label (`SUPPORT`, `ATTACK`, `NONE`)
5. Export the completed annotation as a JSON file

**Output format (`Argument_Relation/<doc_id>.json`):**
```json
{
  "document_id": "Alzheimer_Disease_Tests",
  "relations": [
    {
      "source": 14,
      "target": 2,
      "source_text": "Studies show PET imaging...",
      "target_text": "Aetna considers Brain PET scans medically necessary...",
      "relation": "SUPPORT",
      "similarity": 0.812
    }
  ]
}
```

---

## Label Definitions

### Component Labels

| Label | Definition |
|---|---|
| `CLAIM` | A sentence asserting a coverage decision (medically necessary / not necessary / experimental) |
| `PREMISE` | A sentence providing clinical evidence or policy criteria justifying a decision |
| `NON_ARGUMENTATIVE` | Administrative, procedural, or bibliographic content not part of the argument structure |

### Relation Labels

| Label | Definition |
|---|---|
| `SUPPORT` | The premise provides justification for the claim |
| `ATTACK` | The premise contradicts or undermines the claim |
| `NONE` | No argumentative relation between the pair |

---

## Dependencies

```bash
pip install sentence-transformers spacy
python -m spacy download en_core_web_sm
```
