"""
docuseek/query/ner.py
----------------------
NER-based query enrichment using GLiNER.

Extracts named entities (library names, class names, concepts) from
the query and annotates them inline with their type.  This helps the
retriever match documentation pages that mention the entity by name
but don't repeat the user's phrasing.

Example::

    → "How do I use LoRA with PEFT?"
    → "How do I use {LoRA}{concept} with {PEFT}{library}?"
"""

import structlog
from gliner import GLiNER

from docuseek.config import settings

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
    """Enrich queries by annotating extracted named entities inline."""

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
        entities = self._gliner.predict_entities(
            query,
            _LABELS,
            threshold=self._threshold,
        )

        if not entities:
            return [query]

        rewritten = query
        for entity in entities:
            rewritten = rewritten.replace(
                entity["text"],
                f"{{{entity['text']}}}{{{entity['label']}}}",
            )

        logger.debug(
            "ner_rewrite",
            original=query,
            rewritten=rewritten,
            entities=len(entities),
        )

        return [rewritten]
