# LLM-Litmus Dataset Schema

This document describes the evaluation dataset schema used by LLM-Litmus.

Any team can create their own evaluation dataset for their domain by following this schema.
Save it as a CSV and pass it via `--dataset` flag or the `DATASET_PATH` env variable.

---

## CSV Column Reference

| Column | Type | Required | Description |
|---|---|---|---|
| `unique_id` | string | ✅ | Stable identifier. Use a consistent prefix e.g. `Q001`, `LEGAL001`, `HR042` |
| `question` | string | ✅ | The benchmark query to evaluate |
| `ground_truth` | string | ✅ | The correct expected answer |
| `category` | string | ✅ | One of: `factual` `reasoning` `multi_hop` `edge_case` `out_of_scope` `adversarial` |
| `difficulty` | string | ✅ | One of: `easy` `medium` `hard` |
| `expected_sources` | string | ✅ | Semicolon-separated filenames that should be retrieved (e.g. `doc1.txt;doc2.txt`). Use `N/A` if not applicable |
| `expected_citations` | string | ⬜ | A short verbatim phrase from the source doc that supports the answer. Used for context recall scoring |
| `tags` | string | ⬜ | Comma-separated topic tags (e.g. `deductions, limits, eligibility`) |
| `reasoning_type` | string | ⬜ | One of: `direct_lookup` `multi_step` `comparative` `negation` `numerical` |
| `version` | string | ⬜ | Dataset version for tracking changes (e.g. `1.0`) |
| `evaluation_notes` | string | ⬜ | Human-annotated notes for tricky or ambiguous questions |
| `adversarial_category` | string | ⬜ | Only for adversarial datasets. One of: `prompt_injection` `hallucination_trap` `negation_question` `missing_context` `misleading_retrieval` `numerical_precision` `conflicting_docs` `instruction_override` |

---

## Category Definitions

| Category | When to use |
|---|---|
| `factual` | Direct fact retrieval — single clear answer in one document |
| `reasoning` | Requires applying a rule or condition from context |
| `multi_hop` | Answer requires combining information from multiple documents |
| `edge_case` | Boundary conditions, eligibility thresholds, special cases |
| `out_of_scope` | Question about a topic not in any document — model should refuse gracefully |
| `adversarial` | Deliberately tricky or hostile inputs testing robustness |

---

## Reasoning Type Definitions

| Type | Example |
|---|---|
| `direct_lookup` | "What is the Section 80C deduction limit?" |
| `multi_step` | "If I earn X and invest Y in PPF, what is my net taxable income?" |
| `comparative` | "Which provides more benefit: 80C or 80CCD(1B)?" |
| `negation` | "Is it true that you CANNOT claim HRA if working from home?" |
| `numerical` | "Calculate the exact TDS deductible on a salary of ₹8.5 lakh" |

---

## Example Row

```csv
unique_id,question,ground_truth,category,difficulty,expected_sources,expected_citations,tags,reasoning_type,version,evaluation_notes
Q001,"What is the maximum deduction under Section 80C?","The maximum deduction under Section 80C is ₹1.5 lakh (₹1,50,000) per financial year.",factual,easy,section_80c_deductions.txt,"The aggregate deduction under Section 80C is limited to ₹1,50,000","80c,deduction,limit",direct_lookup,1.0,""
```

---

## Creating Your Own Dataset

### Option 1: Manual authoring
Write questions and ground truths by hand based on your documents.
High quality — recommended for critical evaluation benchmarks.

### Option 2: Synthetic generation
Use the generator to synthesize questions from your corpus:

```bash
# For a legal knowledge base
EVAL_DOMAIN=legal \
EVAL_DOMAIN_DESCRIPTION="Legal Q&A knowledge base" \
DOCS_FOLDER=./legal_docs \
python generate_dataset.py \
  --output ./legal_dataset.csv \
  --target-total 100

# Check diversity and category balance
python generate_dataset.py --output ./legal_dataset.csv  # runs validation report only
```

### Option 3: Hybrid
Manually author difficult edge cases and adversarial questions.
Synthesize the standard factual and reasoning questions.
Combine both into one CSV.

---

## Running Evaluation on Your Dataset

```bash
# Evaluate a legal RAG system
EVAL_DOMAIN=legal \
EVAL_DOMAIN_DESCRIPTION="Legal Q&A knowledge base" \
python evaluate.py \
  --docs-folder ./legal_docs \
  --dataset ./legal_dataset.csv \
  --smoke

# Evaluate a healthcare RAG system
EVAL_DOMAIN=healthcare \
EVAL_DOMAIN_DESCRIPTION="Clinical guidelines and healthcare policy knowledge base" \
python evaluate.py \
  --docs-folder ./clinical_docs \
  --dataset ./clinical_dataset.csv
```

---

## Example Datasets

| File | Domain | Questions |
|---|---|---|
| `examples/tax_golden_dataset.csv` | Indian income tax | 204 |
| `adversarial/adversarial_dataset.csv` | Tax (adversarial) | 15 |
