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
    """
    Retrieves chunks using dense vector search via Qdrant.

    Attributes:
        _client:          Connected QdrantClient instance.
        _collection_name: Qdrant collection to query.
        _embedder:        DenseEmbedder dense embedding model for query encoding.
    """

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
        Embed the query and return the top_k most similar chunks from Qdrant.

        Args:
            query: Raw query string from the user.
            top_k: Number of chunks to return.

        Returns:
            List of Chunk objects ordered by descending similarity score.
        """
        query_embd = self._embedder.embed_query(query)
        results = self._client.query_points(
            collection_name=self._collection_name,
            query=query_embd,
            using=settings.dense_embd_model_name,
            with_payload=True,
            limit=top_k,
        )

        return [
            Chunk(**{k: v for k, v in result.payload.items() if k != "chunk_id"})
            for result in results.points
        ]

    def retrieve_timed(
        self, query: str, top_k: int = settings.retrieval_top_k
    ) -> tuple[list[Chunk], LatencySample]:
        """
        Embed the query, search Qdrant, and return per-component latency.

        Args:
            query: Raw query string from the user.
            top_k: Number of chunks to return.

        Returns:
            Chunks matching ``retrieve``, plus a ``LatencySample`` with
            encoding_ms and search_ms measured independently.
        """
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
