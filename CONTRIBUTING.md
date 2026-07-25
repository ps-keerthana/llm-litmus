# Contributing to LLM-Litmus

Thank you for your interest in contributing! This guide covers development setup, project conventions, and how to submit changes.

---

## Development setup

### 1. Fork and clone

```bash
git clone https://github.com/<your-username>/llm-litmus.git
cd llm-eval-pipeline
```

### 2. Create a virtual environment

```bash
python -m venv venv
.\venv\Scripts\activate       # Windows
# or: source venv/bin/activate  # macOS / Linux
```

### 3. Install dependencies

```bash
# Core + dashboard + API
pip install -r requirements.txt

# Dev tools (testing + coverage)
pip install pytest pytest-cov
```

### 4. Set up your `.env` file

```env
GROQ_API_KEY=gsk_your_key_here
```

---

## Running tests

```bash
# Run all tests
python -m pytest tests/ -v

# Run a specific test file
python -m pytest tests/test_metrics.py -v

# Run with line coverage
python -m pytest tests/ --cov=core --cov-report=term-missing
```

All tests in `tests/` should pass without an API key. Tests that require live LLM calls are marked as integration tests and excluded from the default `pytest` run.

---

## Project conventions

### Code style

- **Python**: PEP 8 compatible. Use clear, descriptive names.
- **Docstrings**: Every public function must have a docstring explaining what it does, its parameters, and its return value.
- **Type hints**: Use type hints on all function signatures.
- **Comments**: Inline comments should explain *why*, not *what*.

### Commit messages

Use imperative present tense:

```
fix(judge): correct auto-fail faithfulness from 1.0 to 0.0
feat(retrieval): add MAP deduplication for repeated sources
docs(readme): update pass rate to latest benchmark result
test(metrics): add numeric extraction edge case for crore suffix
```

### Branch naming

```
fix/auto-fail-faithfulness-logic
feat/chunk-overlap-config
docs/readme-architecture-diagram
test/metrics-negation-edge-cases
```

---

## Pull request guidelines

1. **Open an issue first** for non-trivial changes so we can discuss the approach before implementation.
2. **One logical change per PR** — keep diffs focused and reviewable.
3. **Add or update tests** for any changes to `core/` logic.
4. **Update `CHANGELOG.md`** with a short description of what changed and why.
5. **Do not commit**: `.env`, `eval_platform.db`, `eval_results/`, or `venv/`.

---

## Adding a new LLM provider

1. Create `core/providers/<name>.py` implementing the abstract interface in `core/providers/base.py`.
2. Register it in `core/providers/factory.py`.
3. Add the model name to `config.py` and to the `README.md` provider table.
4. Update `docs/` or add a note in `CHANGELOG.md`.

---

## Adding a new knowledge domain

The platform is domain-agnostic. To evaluate a new corpus:

1. Create a folder for your documents (e.g. `my_docs/`).
2. Set `DOCS_FOLDER=my_docs` and `EVAL_DOMAIN=my_domain` in your `.env`.
3. Generate a benchmark dataset: `python generate_dataset.py --docs-folder my_docs`.
4. Run: `python evaluate.py --docs-folder my_docs --dataset my_dataset.csv`.

---

## Reporting bugs

Open a GitHub Issue with:

- Python version and OS
- The exact command you ran
- The full error traceback
- Which provider you were using

---

## Questions?

Open a [GitHub Discussion](https://github.com/ps-keerthana/llm-litmus/discussions) for questions, ideas, or feedback.
