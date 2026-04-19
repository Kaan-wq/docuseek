"""
docuseek/generation/base.py
----------------------------
Protocol defining the interface for all generation strategies.
"""

from typing import Protocol

from docuseek.chunking.base import Chunk


class BaseGenerator(Protocol):
    def generate(self, query: str, chunks: list[Chunk]) -> str:
        """
        Generate an answer grounded in the retrieved chunks.

        Args:
            query:  Raw query string from the user.
            chunks: Retrieved chunks to use as context.

        Returns:
            Generated answer string.
        """
        ...
