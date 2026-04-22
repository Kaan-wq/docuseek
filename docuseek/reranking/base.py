"""
docuseek/reranking/base.py
---------------------------
Protocol defining the interface for all reranking strategies.

Rerankers are the second stage of the two-stage retrieval pipeline:

    Stage 1: BM25 + dense hybrid (RRF) → top 50-100 candidates  (fast, high recall)
    Stage 2: reranker                  → top 10-15              (slow, high precision)

Every reranker accepts a query and a list of candidate chunks from
stage 1, then returns a smaller list reordered by relevance.
"""

from typing import Protocol

from docuseek.chunking.base import Chunk


class BaseReranker(Protocol):
    def rerank(self, query: str, chunks: list[Chunk], top_k: int = 10) -> list[Chunk]:
        """
        Reorder and truncate candidate chunks by relevance to the query.

        Args:
            query:  Raw query string from the user.
            chunks: Candidate chunks from first-stage retrieval, already
                    roughly ordered but not yet precisely scored.
            top_k:  Number of chunks to return after reranking.

        Returns:
            The top_k most relevant chunks, ordered by descending
            reranker score.
        """
        ...
