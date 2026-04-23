"""
docuseek/chunking/semantic.py
------------------------------
Semantic chunker: splits on meaning shifts rather than character count.

Delegates to chonkie's SemanticChunker, which embeds each sentence,
computes cosine similarity between adjacent sentences within a sliding
window, and inserts a boundary where similarity drops below a threshold.
"""

import chonkie

from docuseek.chunking.base import Chunk
from docuseek.ingestion.cleaners import CleanDocument


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
        """
        Args:
            embedder:       sentence-transformers model name used for
                            sentence similarity. Should match the dense
                            embedder used for indexing so representations
                            are consistent.
            threshold:      Cosine similarity below which a boundary is
                            inserted. Lower values produce larger chunks;
                            higher values split more aggressively.
            min_chunk_size: Minimum number of characters per sentence
                            before a boundary is considered. Guards against
                            very short chunks from isolated heading lines.
            max_chunk_size: Hard character ceiling. Forces a split
                            regardless of similarity when exceeded.
            window_size:    Number of adjacent sentences considered when
                            computing similarity at each candidate boundary.
        """

        self._chunker = chonkie.SemanticChunker(
            embedding_model=embedder,
            threshold=threshold,
            chunk_size=max_chunk_size,
            min_characters_per_sentence=min_chunk_size,
            similarity_window=window_size,
        )

    def chunk(self, doc: CleanDocument) -> list[Chunk]:
        """
        Split a CleanDocument into semantically coherent chunks.

        Args:
            doc: Cleaned document from the ingestion pipeline.

        Returns:
            List of Chunk objects in document order.
        """

        chunks = self._chunker.chunk(doc.content)
        return [
            Chunk(
                content=chunk.text,
                doc_url=doc.url,
                doc_title=doc.title,
                source=doc.source,
                chunk_index=i,
                chunk_total=len(chunks),
                metadata={**doc.metadata},
            )
            for i, chunk in enumerate(chunks)
        ]
