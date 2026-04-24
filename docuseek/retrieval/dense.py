"""
docuseek/retrieval/dense.py
----------------------------
Dense retriever: embeds the query and searches Qdrant by cosine similarity.
"""

import time

from qdrant_client import QdrantClient

from docuseek.chunking.base import Chunk
from docuseek.config import settings
from docuseek.embedding.dense import DenseEmbedder
from docuseek.eval.latency import LatencySample


class DenseRetriever:
    """Retrieves chunks using dense vector search via Qdrant."""

    def __init__(
        self,
        embedder: DenseEmbedder,
        collection_name: str = settings.qdrant_collection_name,
    ) -> None:
        """
        Args:
            embedder:        DenseEmbedder instance used to encode queries.
            collection_name: Qdrant collection to search against.
        """
        self._embedder = embedder
        self._collection_name = collection_name

        if settings.qdrant_cluster_endpoint:
            self._client = QdrantClient(
                url=settings.qdrant_cluster_endpoint,
                api_key=settings.qdrant_api_key,
            )
        else:
            self._client = QdrantClient(url=f"http://{settings.qdrant_host}:{settings.qdrant_port}")

    def retrieve(self, query: str, top_k: int = settings.retrieval_top_k) -> list[Chunk]:
        """
        Embed the query and return the top-k most similar chunks.

        Latency is measured internally but discarded. Use ``retrieve_timed``
        when per-component timing is needed.

        Args:
            query: Raw query string from the user.
            top_k: Number of chunks to return.

        Returns:
            Chunks ordered by descending similarity score.
        """
        chunks, _ = self._retrieve_inner(query, top_k)
        return chunks

    def retrieve_timed(
        self, query: str, top_k: int = settings.retrieval_top_k
    ) -> tuple[list[Chunk], LatencySample]:
        """
        Embed the query, search Qdrant, and return per-component latency.

        Args:
            query: Raw query string from the user.
            top_k: Number of chunks to return.

        Returns:
            A ``(chunks, latency)`` tuple. ``latency`` breaks down wall-clock time
            into ``encoding_ms`` (query embedding) and ``search_ms``
            (Qdrant round-trip), measured independently.
        """
        return self._retrieve_inner(query, top_k)

    def _retrieve_inner(self, query: str, top_k: int) -> tuple[list[Chunk], LatencySample]:
        t0 = time.perf_counter()
        query_embd = self._embedder.embed_query(query)
        encoding_ms = (time.perf_counter() - t0) * 1000

        t1 = time.perf_counter()
        results = self._client.query_points(
            collection_name=self._collection_name,
            query=query_embd,
            using=settings.dense_embd_model_name,
            with_payload=True,
            limit=top_k,
        )
        search_ms = (time.perf_counter() - t1) * 1000

        chunks = [
            Chunk(**{k: v for k, v in result.payload.items() if k != "chunk_id"})
            for result in results.points
        ]

        return chunks, LatencySample(encoding_ms=encoding_ms, search_ms=search_ms)
