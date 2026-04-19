"""
docuseek/retrieval/base.py
--------------------------
Protocol defining the interface for all retrieval strategies.
"""

from typing import Protocol

from docuseek.chunking.base import Chunk


class BaseRetriever(Protocol):
    def retrieve(self, query: str, top_k: int = ...) -> list[Chunk]:
        """
        Retrieve the most relevant chunks for a given query.

        Args:
            query: Raw query string from the user.
            top_k: Number of chunks to return.

        Returns:
            List of Chunk objects ordered by relevance, most relevant first.
        """
        ...
