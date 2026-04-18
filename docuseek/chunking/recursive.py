"""
docuseek/chunking/recursive.py
------------------------------
Recursive character chunker: tries a hierarchy of separators in order,
falling back to the next one when the current separator produces chunks
that are still too large.

Separator priority (coarse → fine):
  \\n\\n  →  paragraph break
  \\n     →  line break
  " "     →  word boundary
  "."     →  sentence end
  ","     →  clause boundary
  [Unicode punctuation]             (CJK support)
  ""      →  character-level fallback (last resort)

This strategy preserves semantic coherence better than FixedSizeChunker
because it avoids cutting inside paragraphs or sentences when possible.
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter

from docuseek.chunking.base import BaseChunker, Chunk
from docuseek.config import settings
from docuseek.ingestion.cleaners import CleanDocument

_SEPARATORS: list[str] = [
    "\n\n",
    "\n",
    " ",
    ".",
    ",",
    "\u200b",  # zero-width space (common in web-scraped text)
    "\uff0c",  # fullwidth comma (CJK)
    "\u3001",  # ideographic comma (CJK)
    "\uff0e",  # fullwidth full stop (CJK)
    "\u3002",  # ideographic full stop (CJK)
    "",  # character-level fallback
]


class RecursiveChunker:
    """
    Splits a document using a cascade of separators, from coarsest
    (paragraph) to finest (character), to produce chunks that respect
    natural language boundaries as much as possible.

    Attributes:
        _splitter: Underlying LangChain RecursiveCharacterTextSplitter.
    """

    def __init__(
        self,
        chunk_size: int = settings.chunk_size,
        overlap: int = settings.chunk_overlap,
    ) -> None:
        """
        Args:
            chunk_size: Target maximum characters per chunk.
            overlap:    Character overlap between consecutive chunks.
        """
        self._splitter = RecursiveCharacterTextSplitter(
            separators=_SEPARATORS,
            chunk_size=chunk_size,
            chunk_overlap=overlap,
        )

    def chunk(self, doc: CleanDocument) -> list[Chunk]:
        """
        Split a CleanDocument using recursive separator fallback.

        Args:
            doc: Cleaned document from the ingestion pipeline.

        Returns:
            List of Chunk objects ordered as they appear in the document.
        """
        texts: list[str] = self._splitter.split_text(doc.content)
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


_: BaseChunker = RecursiveChunker()
