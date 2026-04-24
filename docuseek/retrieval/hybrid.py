"""
docuseek/retrieval/hybrid.py
-----------------------------
Hybrid retriever combining dense and BM25 retrieval via Reciprocal Rank Fusion (RRF).
"""

from __future__ import annotations

import time

from docuseek.chunking.base import Chunk
from docuseek.eval.latency import LatencySample
from docuseek.retrieval.bm25 import BM25Retriever
from docuseek.retrieval.dense import DenseRetriever


class HybridRetriever:
    """
    Fuses dense and BM25 rankings via Reciprocal Rank Fusion.

    Each retriever fetches ``top_k * oversample`` candidates; oversampling
    ensures the fusion has enough overlap to produce a stable top-k ranking.
    """

    def __init__(
        self,
        dense: DenseRetriever,
        bm25: BM25Retriever,
        rrf_k: int = 60,
        oversample: int = 2,
    ) -> None:
        """
        Args:
            dense:      Configured DenseRetriever instance.
            bm25:       Configured BM25Retriever instance.
            rrf_k:      RRF ranking constant.
            oversample: Candidate multiplier applied to top_k before fusion.
        """
        self._dense = dense
        self._bm25 = bm25
        self._rrf_k = rrf_k
        self._oversample = oversample

    def retrieve(self, query: str, top_k: int) -> list[Chunk]:
        """
        Retrieve and fuse results from both retrievers.

        Latency is measured internally but discarded. Use ``retrieve_timed``
        when per-component timing is needed.

        Args:
            query: Natural language query string.
            top_k: Number of results to return after fusion.

        Returns:
            Chunks ordered by RRF score, best first.
        """
        chunks, _ = self._retrieve_inner(query, top_k)
        return chunks

    def retrieve_timed(self, query: str, top_k: int) -> tuple[list[Chunk], LatencySample]:
        """
        Retrieve and fuse results from both retrievers, returning per-component latency.

        ``encoding_ms`` is the sum of both encoding steps (sequential).
        ``search_ms`` is the sum of both Qdrant round-trips plus RRF fusion overhead.

        Args:
            query: Natural language query string.
            top_k: Number of results to return after fusion.

        Returns:
            A ``(chunks, latency)`` tuple with encoding and search durations.
        """
        return self._retrieve_inner(query, top_k)

    def _retrieve_inner(self, query: str, top_k: int) -> tuple[list[Chunk], LatencySample]:
        candidates = top_k * self._oversample

        dense_chunks, dense_sample = self._dense.retrieve_timed(query, candidates)
        bm25_chunks, bm25_sample = self._bm25.retrieve_timed(query, candidates)

        t0 = time.perf_counter()
        fused = self._rrf(dense_chunks, bm25_chunks)[:top_k]
        rrf_ms = (time.perf_counter() - t0) * 1000

        return fused, LatencySample(
            encoding_ms=dense_sample.encoding_ms + bm25_sample.encoding_ms,
            search_ms=dense_sample.search_ms + bm25_sample.search_ms + rrf_ms,
        )

    def _rrf(self, dense_results: list[Chunk], bm25_results: list[Chunk]) -> list[Chunk]:
        """
        Fuse two ranked lists using Reciprocal Rank Fusion.

        Each chunk is scored as the sum of ``1 / (rrf_k + rank)`` across both lists,
        where rank is 1-indexed. Chunks appearing in only one list are scored from
        that list alone.
        """
        scores: dict[str, tuple[Chunk, float]] = {}

        for result_list in [dense_results, bm25_results]:
            for i, chunk in enumerate(result_list, start=1):
                key = str(chunk.chunk_id)
                current = scores.get(key, (chunk, 0.0))
                scores[key] = (current[0], current[1] + 1 / (self._rrf_k + i))

        return [chunk for chunk, _ in sorted(scores.values(), key=lambda x: x[1], reverse=True)]
