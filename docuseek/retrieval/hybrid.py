"""
docuseek/retrieval/hybrid.py
-----------------------------
Hybrid retriever combining dense and sparse BM25 retrieval via
Reciprocal Rank Fusion (RRF).

Runs dense and BM25 retrieval independently, then fuses the ranked
result lists using RRF to produce a single merged ranking. RRF is
robust to score scale differences between retrievers — it operates
on ranks, not raw scores, so no normalisation is needed.

Reference: Cormack et al., "Reciprocal Rank Fusion outperforms Condorcet
and individual Rank Learning Methods" (SIGIR 2009).
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

    Retrieves top_k * rrf_oversample candidates from each retriever,
    fuses them, and returns the top_k results. Oversampling ensures
    the fusion has enough candidates to produce a stable top_k ranking
    even when the two retrievers have low overlap.

    Attributes:
        _dense:        Dense retriever instance.
        _bm25:         BM25 sparse retriever instance.
        _rrf_k:        RRF constant. Controls influence of high-ranked
                       documents. 60 is the standard default from the
                       original paper.
        _oversample:   Multiplier applied to top_k when querying each
                       retriever to ensure sufficient fusion candidates.
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
            rrf_k:      RRF constant.
            oversample: Multiplier on top_k for candidate retrieval.
        """
        self._dense = dense
        self._bm25 = bm25
        self._rrf_k = rrf_k
        self._oversample = oversample

    def retrieve(self, query: str, top_k: int) -> list[Chunk]:
        """
        Retrieve and fuse results from dense and BM25 retrievers.

        Args:
            query: Natural language query string.
            top_k: Number of results to return after fusion.

        Returns:
            List of Chunk objects ordered by RRF score, best first.
        """
        dense_results = self._dense.retrieve(query, top_k * self._oversample)
        bm25_results = self._bm25.retrieve(query, top_k * self._oversample)
        return self._rrf(dense_results, bm25_results)[:top_k]

    def _rrf(
        self,
        dense_results: list[Chunk],
        bm25_results: list[Chunk],
    ) -> list[Chunk]:
        """
        Fuse two ranked lists using Reciprocal Rank Fusion.

        Scores each chunk as sum of 1/(rrf_k + rank) across both lists,
        where rank is 1-indexed. Chunks appearing in only one list still
        receive a score from that list alone.

        Args:
            dense_results: Chunks ranked by dense retriever, best first.
            bm25_results:  Chunks ranked by BM25 retriever, best first.

        Returns:
            Merged list of unique Chunks ordered by RRF score, best first.
        """
        scores: dict[str, tuple[Chunk, float]] = {}

        for result_list in [dense_results, bm25_results]:
            for i, chunk in enumerate(result_list, start=1):
                key = str(chunk.chunk_id)
                if key not in scores:
                    scores[key] = (chunk, 1 / (self._rrf_k + i))
                else:
                    scores[key] = (chunk, scores[key][1] + 1 / (self._rrf_k + i))

        sorted_chunks = sorted(scores.values(), key=lambda x: x[1], reverse=True)
        return [chunk for chunk, _ in sorted_chunks]

    def retrieve_timed(self, query: str, top_k: int) -> tuple[list[Chunk], LatencySample]:
        """
        Retrieve and fuse results from both retrievers, returning per-component latency.

        Delegates timing to each sub-retriever, then times RRF fusion separately.
        encoding_ms is the sum of both encoding steps (they run sequentially).
        search_ms is the sum of both Qdrant queries plus RRF fusion overhead.

        Args:
            query: Natural language query string.
            top_k: Number of results to return after fusion.

        Returns:
            Fused chunks matching ``retrieve``, plus a ``LatencySample``.
        """
        dense_chunks, dense_sample = self._dense.retrieve_timed(query, top_k * self._oversample)
        bm25_chunks, bm25_sample = self._bm25.retrieve_timed(query, top_k * self._oversample)

        t0 = time.perf_counter()
        fused = self._rrf(dense_chunks, bm25_chunks)[:top_k]
        rrf_ms = (time.perf_counter() - t0) * 1000

        return fused, LatencySample(
            encoding_ms=dense_sample.encoding_ms + bm25_sample.encoding_ms,
            search_ms=dense_sample.search_ms + bm25_sample.search_ms + rrf_ms,
        )
