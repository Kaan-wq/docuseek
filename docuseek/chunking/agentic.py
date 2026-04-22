"""
Agentic chunker: delegates boundary decisions to an LLM.

The LLM reads a sliding window of the document and decides whether
the current position represents a meaningful topic boundary. Use
only when benchmarking justifies the cost over semantic chunking.
"""

from docuseek.chunking.base import Chunk
from docuseek.ingestion.cleaners import CleanDocument


class AgenticChunker:
    """
    Uses an LLM to identify topic boundaries in a document.

    Attributes:
        _model:       Model identifier, fixed in settings.
        _window_size: Number of sentences visible to the LLM at each step.
        _max_size:    Hard ceiling — forces a split regardless of LLM output.
    """

    def __init__(
        self,
        window_size: int = 5,
        max_chunk_size: int = 500,
    ) -> None: ...

    def chunk(self, doc: CleanDocument) -> list[Chunk]: ...

    def _is_boundary(self, window: str) -> bool:
        """Ask the LLM whether the centre of this window is a topic boundary."""

    def _build_prompt(self, window: str) -> str: ...
