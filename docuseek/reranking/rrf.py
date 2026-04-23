"""
docuseek/reranking/rrf.py
--------------------------
Reciprocal Rank Fusion for merging ranked lists from multiple query variants.
"""

from __future__ import annotations

from docuseek.chunking.base import Chunk


def reciprocal_rank_fusion(
    ranked_lists: list[list[Chunk]],
    top_k: int | None = None,
    rrf_k: int = 60,
) -> list[Chunk]:
    """Merge multiple ranked chunk lists via Reciprocal Rank Fusion.

    Args:
        ranked_lists: Each inner list is a ranked retrieval result
                      for one query variant, best-first.
        top_k:        If set, truncate the merged list to this length.

    Returns:
        Deduplicated list of chunks sorted by descending RRF score.
    """
    scores: dict[str, float] = {}
    chunks: dict[str, Chunk] = {}

    for ranked in ranked_lists:
        for rank, chunk in enumerate(ranked, start=1):
            cid = str(chunk.chunk_id)
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (rrf_k + rank)
            chunks[cid] = chunk

    merged = sorted(chunks.values(), key=lambda c: scores[str(c.chunk_id)], reverse=True)
    return merged[:top_k] if top_k is not None else merged
