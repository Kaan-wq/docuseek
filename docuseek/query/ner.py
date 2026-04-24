"""
docuseek/query/ner.py
----------------------
NER-based query enrichment using GLiNER.

Example::

    → "How do I use LoRA with PEFT?"
    → "How do I use {LoRA}{concept} with {PEFT}{library}?"
"""

import time

import structlog
from gliner import GLiNER

from docuseek.config import settings
from docuseek.eval.query_metrics import QueryMethodSample

logger = structlog.get_logger(__name__)

_LABELS = [
    "library",
    "class",
    "method",
    "concept",
    "framework",
    "model",
    "task",
]


class NERQueryRewriter:
    """Enriches queries by annotating extracted named entities inline."""

    def __init__(
        self,
        model_name: str = settings.gliner_model_name,
        threshold: float = 0.5,
    ) -> None:
        self._gliner = GLiNER.from_pretrained(model_name)
        self._threshold = threshold
        logger.info("gliner_loaded", model=model_name)

    def rewrite(self, query: str) -> list[str]:
        """Annotate entities in the query with their type labels.

        Args:
            query: Raw user question.

        Returns:
            Single-element list with the annotated query, or ``[query]``
            unchanged if no entities are found.
        """
        variants, _ = self._rewrite_inner(query)
        return variants

    def rewrite_timed(self, query: str) -> tuple[list[str], QueryMethodSample]:
        """Annotate entities and return cost metrics."""
        return self._rewrite_inner(query)

    def _rewrite_inner(self, query: str) -> tuple[list[str], QueryMethodSample]:
        t0 = time.perf_counter()
        entities = self._gliner.predict_entities(
            query,
            _LABELS,
            threshold=self._threshold,
        )
        latency_ms = (time.perf_counter() - t0) * 1000

        rewritten = query
        for entity in entities:
            rewritten = rewritten.replace(
                entity["text"],
                f"{{{entity['text']}}}{{{entity['label']}}}",
            )

        if rewritten != query:
            logger.debug(
                "ner_rewrite",
                original=query,
                rewritten=rewritten,
                entities=len(entities),
            )

        variants = (rewritten,) if rewritten != query else ()

        return [rewritten], QueryMethodSample(
            method="ner",
            latency_ms=latency_ms,
            input_tokens=len(query.split()),
            output_tokens=len(entities),
            original=query,
            variants=variants,
        )
