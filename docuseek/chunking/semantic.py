"""
docuseek/chunking/semantic.py
------------------------------
Semantic chunker: splits on meaning shifts rather than character count.

Delegates to chonkie's SemanticChunker, which embeds each sentence,
computes cosine similarity between adjacent sentences within a sliding
window, and inserts a boundary where similarity drops below a threshold.
"""

import chonkie
import torch

from docuseek.chunking.base import Chunk
from docuseek.ingestion.cleaners import CleanDocument

MAX_CHARS = 20_000


class SemanticChunker:
    """
    Splits a document at points where semantic similarity between
    adjacent sentences drops below a threshold.

    Wraps chonkie.SemanticChunker, which handles sentence splitting,
    embedding, and similarity computation internally.

    Attributes:
        _chunker: Underlying chonkie.SemanticChunker instance.
    """

    def __init__(
        self,
        embedder: str,
        threshold: float = 0.75,
        min_chunk_size: int = 100,
        max_chunk_size: int = 500,
        window_size: int = 5,
    ) -> None:
        self._chunker = chonkie.SemanticChunker(
            embedding_model=embedder,
            threshold=threshold,
            chunk_size=max_chunk_size,
            min_characters_per_sentence=min_chunk_size,
            similarity_window=window_size,
        )

    def _split_sections(self, text: str) -> list[str]:
        """Pre-split text into sections under MAX_CHARS on paragraph boundaries."""
        if len(text) <= MAX_CHARS:
            return [text]

        paragraphs = text.split("\n\n")
        sections: list[str] = []
        current: list[str] = []
        current_len = 0

        for para in paragraphs:
            if current_len + len(para) > MAX_CHARS and current:
                sections.append("\n\n".join(current))
                current = []
                current_len = 0
            current.append(para)
            current_len += len(para)

        if current:
            sections.append("\n\n".join(current))

        return sections

    def chunk(self, doc: CleanDocument) -> list[Chunk]:
        """
        Split a CleanDocument into semantically coherent chunks.

        Args:
            doc: Cleaned document from the ingestion pipeline.

        Returns:
            List of Chunk objects in document order.
        """
        sections = self._split_sections(doc.content)

        all_texts: list[str] = []
        for section in sections:
            with torch.no_grad():
                section_chunks = self._chunker.chunk(section)
            all_texts.extend(c.text for c in section_chunks)

        return [
            Chunk(
                content=text,
                doc_url=doc.url,
                doc_title=doc.title,
                source=doc.source,
                chunk_index=i,
                chunk_total=len(all_texts),
                metadata={**doc.metadata},
            )
            for i, text in enumerate(all_texts)
        ]
