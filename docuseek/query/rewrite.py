"""
docuseek/query/rewrite.py
--------------------------
Composes enabled query strategies into a single rewrite pipeline.

Query rewriting sits upstream of retrieval — it transforms the user's
raw question before it hits the retriever.  Strategies are independent
booleans in ``QueryConfig``.

NER is applied first as a sequential transformation.  HyDE and
multi-query then branch independently from the enriched query
and their results are merged:

    NER (sequential) → HyDE + multi-query (parallel fan-out)

When both HyDE and multi-query are enabled, a single LLM instance
is loaded and shared between them to avoid duplicate memory usage.
"""

from __future__ import annotations

import structlog

from docuseek.experiment_config import QueryConfig
from docuseek.query.hyde import HyDEQueryRewriter
from docuseek.query.model import load_query_model
from docuseek.query.multi_query import MultiQueryRewriter
from docuseek.query.ner import NERQueryRewriter

logger = structlog.get_logger(__name__)


class QueryRewritePipeline:
    """Chains enabled query rewriters into a single callable pipeline.

    Models are loaded once at construction time and reused across all
    queries.  If no strategies are enabled, ``rewrite`` returns the
    original query unchanged.

    Usage::

        pipeline = QueryRewritePipeline(config.query)
        queries = pipeline.rewrite("How do I fine-tune with LoRA?")
    """

    def __init__(self, config: QueryConfig) -> None:
        self._ner: NERQueryRewriter | None = None
        self._hyde: HyDEQueryRewriter | None = None
        self._multi_query: MultiQueryRewriter | None = None

        need_hyde = config.hyde
        need_multi = config.multi_query
        model = None
        tokenizer = None

        if need_hyde and need_multi:
            model, tokenizer = load_query_model()

        if config.ner:
            self._ner = NERQueryRewriter()

        if need_hyde:
            self._hyde = HyDEQueryRewriter(model=model, tokenizer=tokenizer)

        if need_multi:
            self._multi_query = MultiQueryRewriter(model=model, tokenizer=tokenizer)

        active = [
            name
            for name, step in [
                ("NER", self._ner),
                ("HyDE", self._hyde),
                ("MultiQuery", self._multi_query),
            ]
            if step is not None
        ]

        logger.info(
            "query_pipeline_built",
            steps=active,
            shared_llm=need_hyde and need_multi,
        )

    @property
    def enabled(self) -> bool:
        """True if at least one rewrite strategy is active."""
        return any((self._ner, self._hyde, self._multi_query))

    def rewrite(self, query: str) -> list[str]:
        """Apply all enabled strategies.

        NER is applied first as a sequential transformation.
        HyDE and multi-query branch independently from the
        (possibly NER-enriched) query, and their results are merged.

        Args:
            query: Raw user question.

        Returns:
            One or more queries for the retriever.
            Returns ``[query]`` unchanged if no strategies are enabled.
        """
        # Sequential: NER enriches the query for downstream steps
        enriched = query
        if self._ner:
            enriched = self._ner.rewrite(query)[0]

        # Parallel: HyDE and multi-query branch from the enriched query
        queries = [enriched]

        if self._hyde:
            queries.extend(self._hyde.rewrite(enriched))

        if self._multi_query:
            queries.extend(self._multi_query.rewrite(enriched))

        # Deduplicate while preserving order
        queries = list(dict.fromkeys(queries))

        if self.enabled:
            logger.debug(
                "query_rewritten",
                original=query,
                rewritten=queries,
            )

        return queries
