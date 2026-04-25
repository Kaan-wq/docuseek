"""
docuseek/api/routes/query.py
-----------------------------
POST /query — full RAG pipeline: rewrite → retrieve → rerank → generate.

All components are read from app.state (populated by the lifespan in main.py).
Nothing is instantiated per-request.
"""

from __future__ import annotations

import time

import structlog
from fastapi import APIRouter, Request, status
from fastapi.responses import JSONResponse

from docuseek.api.schemas import ErrorResponse, QueryRequest, QueryResponse, SourceInfo
from docuseek.reranking.rrf import reciprocal_rank_fusion

router = APIRouter()
logger = structlog.get_logger()


@router.post(
    "/query",
    response_model=QueryResponse,
    responses={503: {"model": ErrorResponse}},
)
def query(request: Request, body: QueryRequest) -> QueryResponse | JSONResponse:
    """Run the full RAG pipeline and return a grounded answer.

    Pipeline:
        1. Query rewriting  — expand the question into variants (optional)
        2. Retrieval        — retrieve top-k chunks per variant
        3. RRF fusion       — merge variant lists when >1 variant
        4. Reranking        — precision scoring on the candidate pool (optional)
        5. Generation       — produce a grounded answer from top chunks

    Args:
        request: FastAPI request carrying app.state components.
        body:    QueryRequest with the user's question.

    Returns:
        QueryResponse with the answer, deduplicated sources, latency, and
        optionally the query variants that were retrieved against.
    """
    if not hasattr(request.app.state, "retriever"):
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"detail": "Pipeline not initialised. Is the server ready?"},
        )

    state = request.app.state
    config = state.config.eval
    k = config.k_primary

    t_start = time.perf_counter()

    # ── 1. Query rewriting ───────────────────────────────────────────────────
    variants = state.query.rewrite(body.question)
    logger.debug("query_variants", n=len(variants), variants=variants)

    # ── 2. Retrieval ─────────────────────────────────────────────────────────
    raw_lists = [state.retriever.retrieve(q, top_k=k) for q in variants]

    # ── 3. RRF fusion ────────────────────────────────────────────────────────
    chunks = reciprocal_rank_fusion(raw_lists, top_k=k) if len(raw_lists) > 1 else raw_lists[0]

    # ── 4. Reranking ─────────────────────────────────────────────────────────
    if state.reranker is not None:
        chunks = state.reranker.rerank(body.question, chunks, top_k=k)

    # ── 5. Generation ────────────────────────────────────────────────────────
    answer = state.generator.generate(body.question, chunks)

    latency_ms = (time.perf_counter() - t_start) * 1000

    # ── Trace ────────────────────────────────────────────────────────────────
    qt = state.tracer.start(
        question=body.question,
        library="unknown",  # no library tag available at serve time
        difficulty="unknown",
    )
    qt.log_query_rewrite(
        original=body.question,
        variants=variants,
        ms=0.0,  # wall-clock already captured in latency_ms above
    )
    qt.log_retrieval(chunks=chunks, ms=latency_ms)
    if state.reranker is not None:
        qt.log_reranking(chunks=chunks, ms=0.0)
    qt.log_generation(answer=answer)

    # ── Sources ──────────────────────────────────────────────────────────────
    seen: set[str] = set()
    sources: list[SourceInfo] = []
    for chunk in chunks:
        if chunk.doc_url in seen:
            continue
        seen.add(chunk.doc_url)
        sources.append(
            SourceInfo(
                title=chunk.doc_title,
                url=chunk.doc_url,
                source=chunk.source,
            )
        )

    return QueryResponse(
        answer=answer,
        sources=sources,
        latency_ms=round(latency_ms, 1),
        query_variants=variants if len(variants) > 1 else None,
    )
