"""
docuseek/chunking/fixed.py
--------------------------
Fixed-size chunker: splits text into chunks of exactly `chunk_size`
characters with a sliding `overlap` window.

This is the Week 1 baseline. It makes no attempt to respect sentence,
paragraph, or section boundaries — it splits purely by character count.
Use it to establish a performance floor; every other strategy should
beat it on retrieval metrics.

Backed by LangChain's CharacterTextSplitter, which splits on a single
separator (newline by default) and then merges pieces up to chunk_size.
"""

from langchain_text_splitters import CharacterTextSplitter

from docuseek.chunking.base import Chunk
from docuseek.config import settings
from docuseek.ingestion.cleaners import CleanDocument


class FixedSizeChunker:
    """
    Splits a document into fixed-size character chunks with overlap.

    Attributes:
        _splitter: Underlying LangChain CharacterTextSplitter instance.
    """

    def __init__(
        self,
        chunk_size: int = settings.chunk_size,
        overlap: int = settings.chunk_overlap,
    ) -> None:
        """
        Args:
            chunk_size: Maximum number of characters per chunk.
                        Defaults to settings.chunk_size.
            overlap:    Number of characters to repeat at the start of each
                        subsequent chunk to preserve local context across
                        boundaries. Defaults to settings.chunk_overlap.
        """
        self._splitter = CharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            separator="\n",
        )

    def chunk(self, doc: CleanDocument) -> list[Chunk]:
        """
        Split a CleanDocument into fixed-size chunks.

        Args:
            doc: Cleaned document from the ingestion pipeline.

        Returns:
            List of Chunk objects. Order matches the original document.
        """
        texts = self._splitter.split_text(doc.content)
        return [
            Chunk(
                content=text,
                doc_url=doc.url,
                doc_title=doc.title,
                source=doc.source,
                chunk_index=i,
                chunk_total=len(texts),
                metadata={**doc.metadata},
            )
            for i, text in enumerate(texts)
        ]
