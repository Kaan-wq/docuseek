"""
docuseek/chunking/document_structure.py
----------------------------------------
Markdown-aware chunker: splits first on header boundaries to preserve
section structure, then applies recursive splitting to enforce chunk_size.

Stage 1 — MarkdownHeaderTextSplitter
    Splits the document at every heading level (H1 to H6). Each resulting
    piece carries header metadata (e.g. {"Header 2": "Quickstart"}).
    Headers are stripped from the content itself (strip_headers=True)
    to avoid polluting the dense embedding with repeated heading text.

Stage 2 — RecursiveCharacterTextSplitter
    Enforces chunk_size on the pieces produced by Stage 1. A single
    H2 section in the transformers docs can exceed 4 000 characters,
    which would overflow a typical context budget. The recursive pass
    cuts oversized sections while still preferring paragraph and line
    boundaries over arbitrary character positions.

Metadata propagation:
    Header metadata from Stage 1 is merged into each final Chunk's
    metadata dict. This lets the retriever surface which section of a
    doc a chunk came from, which is useful for generating citations.
"""

from langchain_core.documents import Document
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

from docuseek.chunking.base import Chunk
from docuseek.ingestion.cleaners import CleanDocument

_HEADERS_TO_SPLIT_ON: list[tuple[str, str]] = [
    ("#", "header_1"),
    ("##", "header_2"),
    ("###", "header_3"),
    ("####", "header_4"),
    ("#####", "header_5"),
    ("######", "header_6"),
]

_SEPARATORS: list[str] = [
    "\n\n",
    "\n",
    " ",
    ".",
    ",",
    "\u200b",
    "\uff0c",
    "\u3001",
    "\uff0e",
    "\u3002",
    "",
]


class MarkdownHeaderChunker:
    """
    Two-stage chunker that respects Markdown document structure.

    Stage 1 splits on headers to preserve section boundaries.
    Stage 2 enforces a character budget via recursive splitting.
    Header context is forwarded into each chunk's metadata.

    Attributes:
        _md_splitter:  LangChain MarkdownHeaderTextSplitter (Stage 1).
        _recursive_splitter: LangChain RecursiveCharacterTextSplitter (Stage 2).
    """

    def __init__(
        self,
        chunk_size: int = 500,
        overlap: int = 50,
    ) -> None:
        """
        Args:
            chunk_size: Maximum characters per final chunk.
            overlap:    Character overlap between consecutive chunks.
        """
        self._md_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=_HEADERS_TO_SPLIT_ON,
            strip_headers=True,  # keep embeddings clean; headers live in metadata
            return_each_line=False,
        )
        self._recursive_splitter = RecursiveCharacterTextSplitter(
            separators=_SEPARATORS,
            chunk_size=chunk_size,
            chunk_overlap=overlap,
        )

    def chunk(self, doc: CleanDocument) -> list[Chunk]:
        """
        Split a CleanDocument using Markdown structure then recursive sizing.

        Args:
            doc: Cleaned document from the ingestion pipeline.

        Returns:
            List of Chunk objects. Each chunk's metadata contains the
            header hierarchy it belongs to (e.g. header_2: "Installation").
        """
        # Stage 1: split on Markdown headers
        header_docs: list[Document] = self._md_splitter.split_text(doc.content)

        # Stage 2: enforce chunk_size while preserving header metadata
        sized_docs: list[Document] = self._recursive_splitter.split_documents(header_docs)

        return [
            Chunk(
                content=lc_doc.page_content,
                doc_url=doc.url,
                doc_title=doc.title,
                source=doc.source,
                chunk_index=i,
                chunk_total=len(sized_docs),
                metadata={
                    **doc.metadata,
                    **lc_doc.metadata,  # ← merges header_1/2/3 keys
                },
            )
            for i, lc_doc in enumerate(sized_docs)
        ]
