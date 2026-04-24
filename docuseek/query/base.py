"""
docuseek/query/base.py
-----------------------
Protocol defining the interface for query transformation strategies.

Query strategies sit upstream of retrieval — they transform the user's
raw question before it hits the retriever.  Each strategy takes a single
query string and returns one or more rewritten queries:

- NER:         enrich the query with extracted entities → 1 query
- HyDE:        generate a hypothetical answer to embed → 1 query
- Multi-query: generate query variants                 → N queries

The retrieval step runs each returned query, merges the results, and
deduplicates.  Strategies are composable: when multiple are enabled,
they are applied in order NER → HyDE → multi-query, each operating
on the output(s) of the previous step.
"""

from typing import Protocol

from docuseek.eval.query_metrics import QueryMethodSample


class BaseQueryRewriter(Protocol):
    """Protocol for all query transformation strategies.

    Every rewriter takes a single raw query and returns a list of
    rewritten queries.  Strategies that produce a single transformed
    query (NER, HyDE) return a one-element list.  Strategies that
    expand the query (multi-query) return multiple elements.
    """

    def rewrite(self, query: str) -> list[str]:
        """Transform a raw user query into one or more retrieval queries.

        Args:
            query: The original user question.

        Returns:
            One or more rewritten queries to run against the retriever.
        """
        ...

    def rewrite_timed(self, query: str) -> tuple[list[str], QueryMethodSample]:
        """Transform a query and return a QueryMethodSample with cost metrics.

        Args:
            query: The original user question.

        Returns:
            Rewritten queries and a sample carrying latency, token counts,
            and the variants produced.
        """
        ...
