"""
docuseek/reranking/cross_encoder.py
------------------------------------
Cross-encoder reranker — highest accuracy, slowest.

Unlike bi-encoders which encode query and document independently,
a cross-encoder takes the (query, document) pair as a single input
and outputs a relevance score directly.  This makes it the most
accurate reranking approach but only feasible on a small candidate set.

Model: cross-encoder/ms-marco-MiniLM-L6-v2
"""

import structlog
from sentence_transformers import CrossEncoder

from docuseek.chunking.base import Chunk

logger = structlog.get_logger(__name__)


class CrossEncoderReranker:
    """Rerank chunks using a cross-encoder relevance model."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L6-v2") -> None:
        self._model_name = model_name
        self._cross_encoder = CrossEncoder(model_name)
        logger.info("cross_encoder_loaded", model=model_name)

    def rerank(self, query: str, chunks: list[Chunk], top_k: int = 10) -> list[Chunk]:
        """Reorder candidate chunks by cross-encoder relevance score.

        Args:
            query:  Raw query string.
            chunks: Candidate chunks from first-stage retrieval.
            top_k:  Number of chunks to return.

        Returns:
            Top-k chunks ordered by descending relevance score.
        """
        if not chunks or len(chunks) <= top_k:
            return chunks

        pairs = [(query, chunk.content) for chunk in chunks]
        scores = self._cross_encoder.predict(pairs)

        scored_chunks = sorted(
            zip(chunks, scores, strict=True),
            key=lambda x: x[1],
            reverse=True,
        )

        logger.debug(
            "cross_encoder_reranked",
            candidates=len(chunks),
            top_k=top_k,
            top_score=float(scored_chunks[0][1]),
        )

        return [chunk for chunk, _ in scored_chunks[:top_k]]
