"""
docuseek/observability/langfuse_tracer.py
------------------------------------------
Langfuse tracing for benchmark runs.

One trace per question, one span per pipeline stage.
Retrieval metrics are attached as numeric scores on the trace.

All Langfuse calls are wrapped in try/except — a cloud outage or
misconfiguration must never abort a benchmark run.

Usage (benchmark.py)
--------------------
    tracer = LangfuseTracer(experiment_name=config.name)
    for question in questions:
        qt = tracer.start(question.question, question.library, question.difficulty)
        qt.log_query_rewrite(original=question.question, variants=variants, ms=elapsed)
        qt.log_retrieval(chunks=chunks, ms=elapsed)
        qt.log_reranking(chunks=reranked, ms=elapsed)       # optional
        qt.log_generation(answer=answer)                    # optional
        qt.log_scores(scores)
    tracer.flush()
"""

from __future__ import annotations

import structlog
from langfuse import Langfuse

from docuseek.chunking.base import Chunk
from docuseek.config import settings

logger = structlog.get_logger()


class QuestionTrace:
    """Thin wrapper around a single Langfuse trace for one benchmark question."""

    def __init__(self, trace: object, enabled: bool) -> None:
        self._trace = trace
        self._enabled = enabled

    def log_query_rewrite(
        self,
        original: str,
        variants: list[str],
        ms: float,
    ) -> None:
        if not self._enabled:
            return
        try:
            self._trace.span(
                name="query_rewrite",
                input={"original": original},
                output={"variants": variants, "n_variants": len(variants)},
                metadata={"latency_ms": ms},
            )
        except Exception:
            logger.warning("langfuse_span_failed", span="query_rewrite")

    def log_retrieval(
        self,
        chunks: list[Chunk],
        ms: float,
    ) -> None:
        if not self._enabled:
            return
        try:
            self._trace.span(
                name="retrieval",
                output={
                    "n_chunks": len(chunks),
                    "urls": list(dict.fromkeys(c.doc_url for c in chunks))[:10],
                },
                metadata={"latency_ms": ms},
            )
        except Exception:
            logger.warning("langfuse_span_failed", span="retrieval")

    def log_reranking(
        self,
        chunks: list[Chunk],
        ms: float,
    ) -> None:
        if not self._enabled:
            return
        try:
            self._trace.span(
                name="reranking",
                output={
                    "n_chunks": len(chunks),
                    "urls": list(dict.fromkeys(c.doc_url for c in chunks)),
                },
                metadata={"latency_ms": ms},
            )
        except Exception:
            logger.warning("langfuse_span_failed", span="reranking")

    def log_generation(self, answer: str) -> None:
        if not self._enabled:
            return
        try:
            self._trace.span(
                name="generation",
                output={"answer": answer},
            )
        except Exception:
            logger.warning("langfuse_span_failed", span="generation")

    def log_scores(self, scores: dict[str, float]) -> None:
        """Attach retrieval metrics as numeric scores on the trace."""
        if not self._enabled:
            return
        for name, value in scores.items():
            try:
                self._trace.score(name=name, value=value)
            except Exception:
                logger.warning("langfuse_score_failed", metric=name)


class LangfuseTracer:
    """Creates per-question traces for a benchmark run.

    Instantiate once per run, call start() for each question,
    call flush() after the loop.
    """

    def __init__(self, experiment_name: str) -> None:
        self._experiment_name = experiment_name
        self._enabled = False
        self._client: Langfuse | None = None

        try:
            self._client = Langfuse(
                public_key=settings.langfuse_public_key,
                secret_key=settings.langfuse_secret_key,
                host=settings.langfuse_host,
            )
            self._enabled = True
            logger.info("langfuse_connected", host=settings.langfuse_host)
        except Exception:
            logger.warning("langfuse_unavailable", reason="init failed — tracing disabled")

    def start(
        self,
        question: str,
        library: str,
        difficulty: str,
    ) -> QuestionTrace:
        """Open a new trace for one benchmark question.

        Args:
            question:   Raw question text (used as trace input).
            library:    Library tag (e.g. "transformers") for filtering in UI.
            difficulty: Difficulty tag (e.g. "hard") for filtering in UI.

        Returns:
            QuestionTrace with log_* methods. All methods are no-ops if
            Langfuse is unavailable.
        """
        if not self._enabled or self._client is None:
            return QuestionTrace(trace=None, enabled=False)

        try:
            trace = self._client.trace(
                name=self._experiment_name,
                input={"question": question},
                tags=[library, difficulty, self._experiment_name],
                metadata={"library": library, "difficulty": difficulty},
            )
            return QuestionTrace(trace=trace, enabled=True)
        except Exception:
            logger.warning("langfuse_trace_failed", question=question[:60])
            return QuestionTrace(trace=None, enabled=False)

    def flush(self) -> None:
        """Flush all pending traces to Langfuse Cloud. Call once after the loop."""
        if not self._enabled or self._client is None:
            return
        try:
            self._client.flush()
            logger.info("langfuse_flushed")
        except Exception:
            logger.warning("langfuse_flush_failed")
