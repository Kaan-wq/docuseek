"""
docuseek/query/rewrite.py
--------------------------
Composes enabled query strategies into a single rewrite pipeline.

NER applied first as sequential transformation. HyDE & multi-query
branch independently from enriched query and results are merged:

    NER (sequential) → HyDE + multi-query (parallel)
"""

from __future__ import annotations

import structlog

from docuseek.eval.query_metrics import QueryMethodSample
from docuseek.experiment_config import QueryConfig
from docuseek.query.hyde import HyDEQueryRewriter
from docuseek.query.model import load_query_model
from docuseek.query.multi_query import MultiQueryRewriter
from docuseek.query.ner import NERQueryRewriter

logger = structlog.get_logger(__name__)


class QueryRewritePipeline:
    """Chains enabled query rewriters into a single callable pipeline.

    Models are loaded once at construction time and reused across all
    queries. If no strategies are enabled, ``rewrite`` returns the
    original query unchanged.

    Usage::

        pipeline = QueryRewritePipeline(config.query)
        queries = pipeline.rewrite("How do I fine-tune with LoRA?")
    """

    def __init__(self, config: QueryConfig) -> None:
        self._ner: NERQueryRewriter | None = None
        self._hyde: HyDEQueryRewriter | None = None
        self._multi_query: MultiQueryRewriter | None = None

        model, tokenizer = None, None
        if config.hyde and config.multi_query:
            model, tokenizer = load_query_model()

        if config.ner:
            self._ner = NERQueryRewriter()
        if config.hyde:
            self._hyde = HyDEQueryRewriter(model=model, tokenizer=tokenizer)
        if config.multi_query:
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
            "query_pipeline_built", steps=active, shared_llm=config.hyde and config.multi_query
        )

    def rewrite(self, query: str) -> list[str]:
        """Apply all enabled strategies.

        NER is applied first as a sequential transformation. HyDE and
        multi-query branch independently from the (possibly NER-enriched)
        query, and their results are merged.

        Args:
            query: Raw user question.

        Returns:
            One or more queries for the retriever. Returns ``[query]``
            unchanged if no strategies are enabled.
        """
        queries, _ = self._rewrite_inner(query)
        return queries

    def rewrite_timed(self, query: str) -> tuple[list[str], dict[str, QueryMethodSample]]:
        """Apply all enabled strategies and return per-method cost samples."""
        return self._rewrite_inner(query)

    def _rewrite_inner(self, query: str) -> tuple[list[str], dict[str, QueryMethodSample]]:
        samples: dict[str, QueryMethodSample] = {}

        enriched = query
        if self._ner:
            ner_result, ner_sample = self._ner.rewrite_timed(query)
            enriched = ner_result[0]
            samples["ner"] = ner_sample

        queries = [enriched]

        if self._hyde:
            hyde_result, hyde_sample = self._hyde.rewrite_timed(enriched)
            queries.extend(hyde_result)
            samples["hyde"] = hyde_sample

        if self._multi_query:
            mq_result, mq_sample = self._multi_query.rewrite_timed(enriched)
            queries.extend(mq_result)
            samples["multi_query"] = mq_sample

        queries = list(dict.fromkeys(queries))

        if samples:
            logger.debug("query_rewritten", original=query, rewritten=queries)

        return queries, samples
