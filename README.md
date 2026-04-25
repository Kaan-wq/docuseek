# DocuSeek

A RAG benchmarking platform for ML framework documentation.

> **Learning project.** Built to consolidate hands-on knowledge of every layer in a production RAG stack: chunking, embedding, retrieval, reranking, query augmentation, generation, and evaluation. The goal was never a shipped product; it was to understand the tradeoffs at each stage through controlled experiments and hard numbers.

---

## What this is

DocuSeek is not a chatbot. It is an **interchangeable-component benchmarking platform** where every design decision is justified by measurement against a fixed evaluation set. Each experiment isolates one variable, produces a `results.json` with NDCG@10 / Recall@10 / Recall@100 / MRR / Precision@10 / Map@10, and is fully reproducible from its `config.yaml`. The corpus is ML framework documentation: HuggingFace Transformers, Diffusers, Datasets, PEFT, Tokenizers, and Accelerate. The gold evaluation set contains 100 questions stratified by library and difficulty (easy / medium / hard).

---

## ⚠️ GPU-only

This project requires a CUDA-capable GPU. It will not run on CPU. The query rewriting models (HyDE, multi-query) and rerankers (ColBERT, cross-encoder) require GPU memory and acceptable latency. All experiments were run on Kaggle GPU notebooks (T4 / P100). The devcontainer is configured for local development only.

---

## Knowledge areas covered

| Area | What was built | Key tradeoff learned |
|---|---|---|
| **Chunking** | Fixed, recursive, markdown-aware, semantic | Semantic chunking improves coherence but is expensive and OOM-prone on large docs. Cost only needs to be paid once! |
| **Embedding** | Dense (Harrier-270m), sparse (BM25) | Both have their advantages, better to combine them! |
| **Retrieval** | Dense-only, BM25-only, hybrid RRF | Surprisingly, BM25 outperforms with clean eval. Hybrid preferred for production when users have messy inputs. |
| **Reranking** | ColBERT (MaxSim), cross-encoder | Cross-encoders can be very fast depending on size and they significantly outperform late interaction models. |
| **Query augmentation** | NER (GLiNER), HyDE, multi-query | These methods are very time-consuming and advantages are marginal. Could still be interesting for messy inputs. |
| **Generation** | Mistral API, CoT, few-shot, budget-forcing | Not tested yet (API cost). |
| **Evaluation** | NDCG@10, Recall@100, MRR, MAP@10, Precision@10 | Evaluation is doc-level (URL-deduplicated), not chunk-level |
| **Observability** | Langfuse Cloud | Full trace visibility: query → variants → retrieved docs → reranked docs → answer |

Late interaction embeddings (ColBERT-style, PLAID index) and Matryoshka quantization are understood theoretically and stubbed for future work.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│  Ingestion pipeline                                     │
│  Raw docs → Chunker → Embedder → Qdrant (dense+sparse)  │
└─────────────────────────────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────┐
│  Query pipeline                                         │
│  Query → [Rewrite] → Retrieve → [RRF] → [Rerank] → LLM  │
└─────────────────────────────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────┐
│  Observability & evaluation                             │
│  Langfuse tracing · NDCG@10 · Recall@100 · benchmark    │
└─────────────────────────────────────────────────────────┘
```

---

## Tech stack

| Component | Choice |
|---|---|
| Vector store | Qdrant (local Docker or Cloud cluster) |
| Dense embedding | `microsoft/harrier-oss-v1-270m` (dim 640) |
| Sparse embedding | `Qdrant/bm25` |
| Reranker (ColBERT) | `colbert-ir/colbertv2.0` |
| Reranker (cross-encoder) | `Alibaba-NLP/gte-reranker-modernbert-base` |
| NER | `urchade/gliner_medium-v2.1` (GLiNER) |
| Query model | `Qwen/Qwen2.5-1.5B-Instruct` |
| Generator | `microsoft/Phi-4-mini-instruct` |
| Observability | Langfuse Cloud |
| API | FastAPI + uvicorn |
| Frontend | Streamlit |
| Package manager | uv |
| Python | 3.12 |

---

## Experiment roadmap

| # | Name | Status | Primary metric |
|---|---|---|---|
| 01 | Chunking strategies | ✅ done | NDCG@10 |
| 02 | Retrieval modes | ✅ done | NDCG@10 |
| 03 | Reranking | ✅ done | NDCG@10 |
| 04 | Query strategies | ✅ done | NDCG@10 |
| 05 | Generation prompting | ⏳ pending | — |
| 06 | Matryoshka quantization | ⏸ deferred | — |

Each experiment folder contains a `config.yaml` (the full resolved config) and a `results.json` (metrics + archived config for full reproducibility).

---

## Repository structure

```
docuseek/
├── ingestion/          # Scrapers
├── chunking/           # Chunkers
├── embedding/          # Dense, sparse
├── retrieval/          # Dense, BM25, hybrid (RRF)
├── reranking/          # ColBERT, cross-encoder
├── query/              # NER, HyDE, multi-query
├── generation/         # Mistral API, prompt assembly
├── eval/               # Evaluation utilities
├── observability/      # Langfuse tracer
└── api/                # FastAPI app (requires GPU)

experiments/            # config.yaml + results.json
scripts/                # ingest, chunk_docs, build_index, benchmark, serve
data/eval/              # gold_set_v1.jsonl (committed)
frontend/               # Streamlit app (requires GPU)
```

---

## Prerequisites

Copy `.env.example` to `.env` and fill in the following secrets:

| Secret | Purpose | Where to get it |
|---|---|---|
| `HF_TOKEN` | Model downloads + faster HF download rates | [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) |
| `GITHUB_TOKEN` | GitHub API access for doc ingestion (scraping) | [github.com/settings/tokens](https://github.com/settings/tokens) |
| `QDRANT_API_KEY` + `QDRANT_CLUSTER_ENDPOINT` | Vector store (if using Cloud) | [cloud.qdrant.io](https://cloud.qdrant.io) |
| `LANGFUSE_HOST` + `LANGFUSE_PUBLIC_KEY` + `LANGFUSE_SECRET_KEY` | Tracing | [cloud.langfuse.com](https://cloud.langfuse.com) |
| `MISTRAL_API_KEY` | Secondary generator (Mistral API backend) | [console.mistral.ai](https://console.mistral.ai) |
| `GEMINI_API_KEY` | Agentic chunker (deferred) | [aistudio.google.com](https://aistudio.google.com) |

> **Qdrant:** a local Docker instance (`docker compose up -d`) is supported for development. For GPU notebooks (Kaggle, Colab) a [Qdrant Cloud](https://cloud.qdrant.io) free-tier cluster is recommended — set `QDRANT_API_KEY` and `QDRANT_CLUSTER_ENDPOINT` accordingly.

---

## Running on Kaggle (recommended)

All experiments were run on Kaggle GPU notebooks (T4 / P100). Add each secret from the table above via *Add-ons → Secrets* in the Kaggle sidebar, then run:

```python
from kaggle_secrets import UserSecretsClient
import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

user_secrets = UserSecretsClient()
for key in [
    "GITHUB_TOKEN", "HF_TOKEN",
    "LANGFUSE_HOST", "LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY",
    "MISTRAL_API_KEY", "GEMINI_API_KEY",
    "QDRANT_API_KEY", "QDRANT_CLUSTER_ENDPOINT",
]:
    os.environ[key] = user_secrets.get_secret(key)

!git clone https://github.com/Kaan-wq/docuseek.git -q
%cd docuseek
!pip install uv -q
!uv sync --all-groups -q
```

Then follow the benchmark steps below.

---

## Running the benchmark

```bash
# 1. (Local only) Start Qdrant — skip if using a Cloud cluster endpoint
docker compose up -d

# 2. Ingest raw documentation
uv run python scripts/ingest.py

# 3. Chunk cleaned documentation
uv run python scripts/chunk_docs.py --config experiments/04_query/none/config.yaml

# 4. Build the index
uv run python scripts/build_index.py --config experiments/04_query/none/config.yaml

# 5. Run the benchmark
uv run python scripts/benchmark.py --config experiments/04_query/none/config.yaml
```
Results are written to `experiments/04_query/none/results.json`.

---

## Running the API (GPU only)

```bash
uv run python scripts/serve.py --config experiments/04_query/none/config.yaml
uv run streamlit run frontend/app.py
```

---

## Deferred / future work

- Matryoshka embeddings at reduced dimensions
- Late interaction retrieval with PLAID index
- Generation evaluation (RAGAS / LLM judge) once budget allows
- PyTorch and vLLM documentation scrapers
- Context-aware chunking
- Tests
- CI/CD pipeline (`ci.yml` scaffolded)

---

## Related

Part of a blog series — *Journey to becoming an LLM Expert*:
- Part 1 ✅ — Baseline GPT from scratch
- Part 2 ○ — Quantization results
- Part 3 ○ — Architecture ablations
- Part 4 ○ — RAG system ← this project