"""
docuseek/retrieval/dense.py
----------------------------
Dense retriever: embeds the query and searches Qdrant by cosine similarity.
"""

from qdrant_client import QdrantClient

from docuseek.chunking.base import Chunk
from docuseek.config import settings
from docuseek.embedding.dense import DenseEmbedder


class DenseRetriever:
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
