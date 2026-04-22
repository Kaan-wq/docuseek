"""
Context-aware wrapper: enriches chunks produced by any chunker with
LLM-generated context before they are embedded.

For each chunk, an LLM receives the full source document and the chunk
and returns a short summary of what the chunk is about in context. This
summary is prepended to the chunk content before embedding.

Reference: Anthropic "Contextual Retrieval" (2024).

This is a wrapper, not a standalone chunker — it delegates boundary
decisions to an inner chunker and post-processes its output.

Usage:
    base = get_chunker(config.chunker)          # any algorithm
    chunker = ContextAwareChunker(inner=base)
    chunks = chunker.chunk(doc)
"""

from docuseek.chunking.base import BaseChunker, Chunk
from docuseek.ingestion.cleaners import CleanDocument


class ContextAwareChunker:
    """
    Wraps any BaseChunker and prepends LLM-generated context to each chunk.

    Attributes:
        _inner: The underlying chunker that determines boundaries.
        _model: Model used for context generation, fixed in settings.
    """

    def __init__(self, inner: BaseChunker) -> None: ...

    def chunk(self, doc: CleanDocument) -> list[Chunk]:
        """
        Chunk the document with the inner chunker, then enrich each chunk.
        The original content is preserved in chunk.metadata["raw_content"].
        """

    def _generate_context(self, doc_content: str, chunk_content: str) -> str:
        """
        Call the LLM to summarise what this chunk is about in the context
        of the full document. Returns a short paragraph to prepend.
        """

    def _build_prompt(self, doc_content: str, chunk_content: str) -> str: ...
