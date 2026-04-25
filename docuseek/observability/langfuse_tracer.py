"""
docuseek/observability/langfuse_tracer.py
------------------------------------------
Langfuse v3 tracing for benchmark runs.

v3 is OTel-based. The API is context-manager driven:
    - client.start_as_current_observation() opens a trace
    - span.start_as_current_observation() opens a child span
    - span.score() attaches numeric scores

One trace per question. Call qt.finish() after log_scores() to close the
trace context. Call tracer.flush() once after the question loop.

All calls wrapped in try/except — a cloud outage never aborts a benchmark run.
"""

from __future__ import annotations

import structlog
from langfuse import Langfuse

from docuseek.chunking.base import Chunk
from docuseek.config import settings

logger = structlog.get_logger()


class QuestionTrace:
    """Wraps a single Langfuse trace for one benchmark question."""

    def __init__(
        self,
        client: Langfuse | None,
        experiment_name: str,
        question: str,
        library: str,
        difficulty: str,
        enabled: bool,
    ) -> None:
        self._enabled = enabled
        self._trace_cm = None
        self._span = None

        if not enabled or client is None:
            return

        try:
            self._trace_cm = client.start_as_current_observation(
                name=experiment_name,
                input={"question": question},
                metadata={"library": library, "difficulty": difficulty},
            )
            self._span = self._trace_cm.__enter__()
        except Exception as e:
            logger.warning("langfuse_trace_failed", question=question[:60], error=str(e))
            self._enabled = False

    def log_query_rewrite(self, original: str, variants: list[str], ms: float) -> None:
        if not self._enabled or self._span is None:
            return
        try:
            with self._span.start_as_current_observation(
                name="query_rewrite",
                input={"original": original},
                output={"variants": variants, "n_variants": len(variants)},
                metadata={"latency_ms": ms},
            ):
                pass
        except Exception as e:
            logger.warning("langfuse_span_failed", span="query_rewrite", error=str(e))

    def log_retrieval(self, chunks: list[Chunk], ms: float) -> None:
        if not self._enabled or self._span is None:
            return
        try:
            with self._span.start_as_current_observation(
                name="retrieval",
                output={
                    "n_chunks": len(chunks),
                    "urls": list(dict.fromkeys(c.doc_url for c in chunks))[:10],
                },
                metadata={"latency_ms": ms},
            ):
                pass
        except Exception as e:
            logger.warning("langfuse_span_failed", span="retrieval", error=str(e))

    def log_reranking(self, chunks: list[Chunk], ms: float) -> None:
        if not self._enabled or self._span is None:
            return
        try:
            with self._span.start_as_current_observation(
                name="reranking",
                output={
                    "n_chunks": len(chunks),
                    "urls": list(dict.fromkeys(c.doc_url for c in chunks)),
                },
                metadata={"latency_ms": ms},
            ):
                pass
        except Exception as e:
            logger.warning("langfuse_span_failed", span="reranking", error=str(e))

    def log_generation(self, answer: str) -> None:
        if not self._enabled or self._span is None:
            return
        try:
            with self._span.start_as_current_observation(
                name="generation",
                output={"answer": answer},
            ):
                pass
        except Exception as e:
            logger.warning("langfuse_span_failed", span="generation", error=str(e))

    def log_scores(self, scores: dict[str, float]) -> None:
        if not self._enabled or self._span is None:
            return
        for name, value in scores.items():
            try:
                self._span.score(name=name, value=value, data_type="NUMERIC")
            except Exception as e:
                logger.warning("langfuse_score_failed", metric=name, error=str(e))

    def finish(self) -> None:
        """Close the trace context. Must be called after log_scores()."""
        if self._trace_cm is not None:
            try:
                self._trace_cm.__exit__(None, None, None)
            except Exception as e:
                logger.warning("langfuse_finish_failed", error=str(e))


class LangfuseTracer:
    """Creates per-question traces for a benchmark run."""

    def __init__(self, experiment_name: str) -> None:
        self._experiment_name = experiment_name
        self._enabled = False
        self._client: Langfuse | None = None

        try:
            self._client = Langfuse(
                public_key=settings.langfuse_public_key,
                secret_key=settings.langfuse_secret_key,
                base_url=settings.langfuse_host,  # host= is deprecated in v3
            )
            self._enabled = True
            logger.info("langfuse_connected", host=settings.langfuse_host)
        except Exception as e:
            logger.warning("langfuse_unavailable", error=str(e))

    def start(self, question: str, library: str, difficulty: str) -> QuestionTrace:
        return QuestionTrace(
            client=self._client,
            experiment_name=self._experiment_name,
            question=question,
            library=library,
            difficulty=difficulty,
            enabled=self._enabled,
        )

    def flush(self) -> None:
        if not self._enabled or self._client is None:
            return
        try:
            self._client.flush()
            logger.info("langfuse_flushed")
        except Exception as e:
            logger.warning("langfuse_flush_failed", error=str(e))
