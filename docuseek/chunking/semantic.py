"""
docuseek/chunking/semantic.py
------------------------------
Semantic chunker: splits on meaning shifts rather than character count.

Delegates to LangChain's SemanticChunker, which embeds each sentence,
computes cosine similarity between adjacent sentences, and inserts a
boundary where similarity exceeds a breakpoint threshold.
"""

from langchain_experimental.text_splitter import SemanticChunker as LangChainSemanticChunker

from docuseek.chunking.base import Chunk
from docuseek.embedding.dense import DenseEmbedder
from docuseek.ingestion.cleaners import CleanDocument


class SemanticChunker:
    """
    Splits a document at points where semantic similarity between
    adjacent sentences drops below a threshold.

    Wraps LangChain's SemanticChunker with DenseEmbedder as the
    embedding backend.

    Attributes:
        _splitter: Underlying LangChain SemanticChunker instance.
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
            threshold:      Breakpoint threshold — sentences above this
                            similarity are merged, below it a boundary
                            is inserted.
            min_chunk_size: Minimum number of characters per chunk.
            max_chunk_size: Unused — kept for interface compatibility.
            window_size:    Unused — kept for interface compatibility.
        """
        self._splitter = LangChainSemanticChunker(
            embeddings=DenseEmbedder(model_name=embedder),
            breakpoint_threshold_type="percentile",
            breakpoint_threshold_amount=threshold,
            min_chunk_size=min_chunk_size,
        )

    def chunk(self, doc: CleanDocument) -> list[Chunk]:
        """
        Split a CleanDocument into semantically coherent chunks.

        Args:
            doc: Cleaned document from the ingestion pipeline.

        Returns:
            List of Chunk objects in document order.
        """

        lc_chunks = self._splitter.split_text(doc.content)
        return [
            Chunk(
                content=text,
                doc_url=doc.url,
                doc_title=doc.title,
                source=doc.source,
                chunk_index=i,
                chunk_total=len(lc_chunks),
                metadata={**doc.metadata},
            )
            for i, text in enumerate(lc_chunks)
        ]
