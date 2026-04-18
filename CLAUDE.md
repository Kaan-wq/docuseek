# DocuSeek — Claude Code working notes

## What this project is

A RAG benchmarking platform for ML framework documentation (HuggingFace, PyTorch, vLLM).
The goal is not to build "one pipeline" — it's to make every RAG component
interchangeable so we can benchmark alternatives on a fixed eval set.

## Architectural principles

- Every major component (chunker, embedder, retriever, generator) is defined by
  a `Protocol` in its `base.py`. Implementations are plug-compatible.
- Benchmarks are configured via YAML in `experiments/XX_name/config.yaml` and
  run via `scripts/run_benchmark.py`. Results go to `experiments/XX_name/results.json`.
- Nothing imports from `experiments/` or `scripts/` — those are entry points, not library code.
- The eval set (`data/eval/gold_set_v1.jsonl`) is the source of truth. All
  design decisions are justified by measurement against it.

## Code conventions

- Python 3.12, `strict` mypy, Ruff with the config in `pyproject.toml`.
- Pydantic v2 for all config, request, and response models.
- `structlog` for logging, never `print`. Scripts may use `rich.print`.
- All I/O is async where it crosses a network boundary (HTTP, DB).
- Tests: unit tests mock external services; integration tests run against
  the full docker-compose stack and are marked `@pytest.mark.integration`.

## Commands

- `uv sync --all-groups` — install deps
- `docker compose up -d` — start Qdrant + Langfuse
- `uv run scripts/ingest.py` — scrape + clean docs
- `uv run scripts/build_index.py --config experiments/01_chunking/config.yaml`
- `uv run scripts/run_benchmark.py --experiment 01_chunking`
- `uv run scripts/serve.py` — start FastAPI on :8000
- `uv run streamlit run frontend/app.py` — start UI on :8501
- `uv run pytest` — run tests
- `uv run pytest -m "not integration"` — fast tests only
- `uv run ruff check . && uv run ruff format .` — lint + format

## When adding a new component

1. Add the Protocol method to the relevant `base.py` if new behavior is needed.
2. Implement in a new file in the same module.
3. Register in the module's `__init__.py` factory if we have one.
4. Add unit tests with a mock dependency.
5. Add it as an option in the relevant experiment YAML.
6. Re-run the benchmark to capture its numbers before committing.

## What NOT to do

- Don't add new dependencies without checking they're maintained and have no
  known security issues. Prefer small, focused libraries.
- Don't hardcode API keys, model names, or paths — everything goes through `config.py`.
- Don't create pipelines that bypass the Protocol abstractions.
- Don't commit anything from `data/raw/` or `data/processed/`. Only `data/eval/`
  is tracked.
- Don't put experimental code in the main package. It lives in `notebooks/` first,
  then graduates to `docuseek/` with tests.

## Current phase

Week 1 of 6. Focus: scaffolding, ingestion pipeline, eval set, baseline RAG.
Do not add advanced features until the baseline is benchmarked.
