"""
docuseek/embedding/base.py
--------------------------
Protocol defining the interface for all embedding strategies.

Two methods are required rather than one because queries and documents
are embedded asymmetrically: queries receive a task instruction prefix
that steers retrieval-tuned models, while documents are encoded as-is.

Concrete classes must implement embed_documents() and embed_queries().
embed_query() is provided as a default convenience wrapper and should
not be overridden.
"""

from abc import ABC, abstractmethod


class BaseEmbedder(ABC):
    @abstractmethod
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """
        Encode a batch of document chunks into dense vectors.
        No instruction prefix is applied.

        Args:
            texts: Chunk content strings to encode.

        Returns:
            List of float vectors, one per input text.
        """
        ...

    @abstractmethod
    def embed_queries(self, queries: list[str]) -> list[list[float]]:
        """
        Encode a batch of query strings into dense vectors.
        An instruction prefix is prepended before encoding.

        Args:
            queries: Raw query strings to encode.

        Returns:
            List of float vectors, one per input query.
        """
        ...

    def embed_query(self, query: str) -> list[float]:
        """
        Convenience wrapper for encoding a single query.
        Delegates to embed_queries() — do not override.

        Args:
            query: Single raw query string.

        Returns:
            Single float vector.
        """
        return self.embed_queries([query])[0]
