"""
docuseek/chunking/base.py
-------------------------
Shared data structures and Protocol for all chunking strategies.

Every chunker in this package implements the BaseChunker Protocol,
meaning it exposes a single `chunk(doc) -> list[Chunk]` method.
"""

import hashlib
from dataclasses import dataclass, field
from typing import Protocol

from docuseek.ingestion.cleaners import CleanDocument


@dataclass
class Chunk:
    """
    A single chunk produced from a CleanDocument.

    Attributes:
        content:     The raw text of this chunk.
        doc_url:     URL of the source document this chunk came from.
        doc_title:   Title of the source document.
        source:      Source identifier, e.g. "huggingface/transformers".
        chunk_index: Zero-based position of this chunk within its document.
        chunk_total: Total number of chunks produced from the source document.
        metadata:    Arbitrary key-value pairs inherited from the source
                     document and optionally extended by the chunker
                     (e.g. which markdown header this chunk falls under).
        chunk_id:    Deterministic 12-character hex ID derived from content.
                     Identical content always produces the same ID, enabling
                     idempotent upserts into Qdrant.
    """

    content: str
    doc_url: str
    doc_title: str
    source: str
    chunk_index: int
    chunk_total: int
    metadata: dict = field(default_factory=dict)
    chunk_id: str = field(init=False)

    def __post_init__(self) -> None:
        self.chunk_id = hashlib.md5(self.content.encode()).hexdigest()[:12]


class BaseChunker(Protocol):
    """
    Protocol that every chunker must satisfy.

    A chunker takes a single CleanDocument and returns a list of Chunk
    objects. The chunker is responsible for splitting the document content,
    assigning chunk indices, and forwarding source metadata.
    """

    def chunk(self, doc: CleanDocument) -> list[Chunk]:
        """
        Split a cleaned document into a list of chunks.

        Args:
            doc: A CleanDocument produced by the ingestion pipeline.

        Returns:
            Ordered list of Chunk objects. May be empty if the document
            content is too short to produce any chunks.
        """
        ...
