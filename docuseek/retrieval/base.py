"""
docuseek/retrieval/base.py
--------------------------
Protocol defining the interface for all retrieval strategies.
"""

from typing import Protocol

from docuseek.chunking.base import Chunk
from docuseek.eval.latency import LatencySample


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

    def retrieve_timed(self, query: str, top_k: int = ...) -> tuple[list[Chunk], LatencySample]:
        """
        Retrieve chunks and return per-component latency alongside results.

        Implementations must time the two steps separately:
        - ``encoding_ms``: query embedding or tokenisation only.
        - ``search_ms``:   Qdrant index query only, from request to response.

        Args:
            query: Raw query string from the user.
            top_k: Number of chunks to return.

        Returns:
            A tuple of (chunks, sample) where chunks is the same ordered list
            as ``retrieve`` would return, and sample is a ``LatencySample``
            with encoding and search times in milliseconds.
        """
        ...
