"""
docuseek/retrieval/bm25.py
---------------------------
Sparse BM25 retriever backed by Qdrant's sparse vector index.

Queries the "bm25" sparse vector field populated during indexing
by FastEmbed's SparseTextEmbedding model. Complements dense retrieval
for exact keyword matching — particularly useful for technical queries
containing precise API names, class names, and method signatures that
semantic embeddings may not surface reliably.
"""

from __future__ import annotations

import time

from fastembed import SparseTextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.models import Document, SparseVector

from docuseek.chunking.base import Chunk
from docuseek.config import settings
from docuseek.eval.latency import LatencySample


class BM25Retriever:
    """
    Retrieves chunks using sparse BM25 vector search via Qdrant.

    Attributes:
        _client:          Connected QdrantClient instance.
        _collection_name: Qdrant collection to query.
        _embedder:        FastEmbed sparse embedding model for query encoding.
    """

    def __init__(
        self,
        collection_name: str = settings.qdrant_collection_name,
    ) -> None:
        """
        Args:
            collection_name: Qdrant collection to query.
        """
        self._embedder = SparseTextEmbedding("Qdrant/bm25")
        self._collection_name = collection_name
        if settings.qdrant_cluster_endpoint:
            self._client = QdrantClient(
                url=settings.qdrant_cluster_endpoint,
                api_key=settings.qdrant_api_key,
            )
        else:
            self._client = QdrantClient(url=f"http://{settings.qdrant_host}:{settings.qdrant_port}")

    def retrieve(self, query: str, top_k: int) -> list[Chunk]:
        """
        Retrieve the top-k chunks most relevant to the query via BM25.

        Args:
            query: Natural language query string.
            top_k: Number of results to return.

        Returns:
            List of Chunk objects ordered by BM25 score, best first.
        """
        results = self._client.query_points(
            collection_name=self._collection_name,
            query=Document(text=query, model=self._embedder),
            using="bm25",
            with_payload=True,
            limit=top_k,
        )

        return [
            Chunk(**{k: v for k, v in result.payload.items() if k != "chunk_id"})
            for result in results.points
        ]

    def retrieve_timed(self, query: str, top_k: int) -> tuple[list[Chunk], LatencySample]:
        """
        Retrieve chunks via BM25 and return per-component latency.

        Encodes query manually before calling Qdrant so the two steps
        can be timed independently.

        Args:
            query: Natural language query string.
            top_k: Number of results to return.

        Returns:
            A tuple of (chunks, sample) where chunks matches what ``retrieve``
            would return, and sample carries encoding_ms and search_ms.
        """
        t0 = time.perf_counter()
        embedding = next(iter(self._embedder.embed([query])))
        encoding_ms = (time.perf_counter() - t0) * 1000

        sparse_vec = SparseVector(
            indices=embedding.indices.tolist(),
            values=embedding.values.tolist(),
        )

        t1 = time.perf_counter()
        results = self._client.query_points(
            collection_name=self._collection_name,
            query=sparse_vec,
            using="bm25",
            with_payload=True,
            limit=top_k,
        )
        search_ms = (time.perf_counter() - t1) * 1000

        chunks = [
            Chunk(**{k: v for k, v in result.payload.items() if k != "chunk_id"})
            for result in results.points
        ]
        return chunks, LatencySample(encoding_ms=encoding_ms, search_ms=search_ms)
