"""
docuseek/chunking/agentic.py
-----------------------------
Agentic chunker: delegates boundary decisions to an LLM.

Uses chonkie's SlumberChunker backed by a Gemini genie to identify
meaningful topic boundaries. The LLM receives a sliding character window
over the document and decides whether the centre of that window represents
a boundary worth splitting on.

Significantly more expensive than semantic chunking — incurs one LLM call
per candidate boundary. Only benchmark this after semantic chunking results
justify the cost.
"""

from chonkie import SlumberChunker
from chonkie.genie import GeminiGenie

from docuseek.chunking.base import Chunk
from docuseek.config import settings
from docuseek.ingestion.cleaners import CleanDocument


class AgenticChunker:
    """
    Uses an LLM to identify topic boundaries in a document.

    Wraps chonkie's SlumberChunker with a GeminiGenie backend.

    Attributes:
        _chunker: Underlying chonkie.SlumberChunker instance.
    """

    def __init__(
        self,
        min_chunk_size: int = 100,
        max_chunk_size: int = 500,
        window_size: int = 128,
    ) -> None:
        """
        Args:
            min_chunk_size: Minimum number of characters per chunk.
                            Guards against degenerate splits on short
                            transitional sentences.
            max_chunk_size: Hard character ceiling. Forces a split
                            regardless of LLM output when exceeded.
            window_size:    Number of tokens in the candidate window
                            passed to the LLM at each boundary decision.
                            Larger values give the model more context but
                            increase token cost per call.
        """

        self._chunker = SlumberChunker(
            genie=GeminiGenie(model=settings.gemini_model, api_key=settings.gemini_api_key),
            tokenizer="gpt2",
            chunk_size=max_chunk_size,
            candidate_size=window_size,
            min_characters_per_chunk=min_chunk_size,
        )

    def chunk(self, doc: CleanDocument) -> list[Chunk]:
        """
        Split a CleanDocument into agentically coherent chunks.

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
