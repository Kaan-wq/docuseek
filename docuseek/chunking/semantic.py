"""
Semantic chunker: splits on meaning shifts rather than character count.

Strategy: embed each sentence, compute cosine similarity between adjacent
sentences, insert a boundary where similarity drops below a threshold.
"""

from docuseek.chunking.base import Chunk
from docuseek.embedding.base import BaseEmbedder
from docuseek.ingestion.cleaners import CleanDocument


class SemanticChunker:
    """
    Splits a document at points where semantic similarity between
    adjacent sentences drops below a threshold.

    Attributes:
        _embedder:  Embedder used to encode sentences.
        _threshold: Cosine similarity below which a boundary is inserted.
        _min_size:  Minimum chunk character length before a boundary is allowed.
        _max_size:  Hard ceiling — forces a split regardless of similarity.
    """

    def __init__(
        self,
        embedder: BaseEmbedder,
        threshold: float = 0.5,
        min_chunk_size: int = 100,
        max_chunk_size: int = 500,
    ) -> None: ...

    def chunk(self, doc: CleanDocument) -> list[Chunk]: ...

    def _split_into_sentences(self, text: str) -> list[str]: ...

    def _cosine_similarity(self, a: list[float], b: list[float]) -> float: ...
