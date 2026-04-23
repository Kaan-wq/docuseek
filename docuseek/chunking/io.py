"""
docuseek/chunking/io.py
-----------------------
Chunk serialisation and file I/O.

Provides the read/write layer for chunk JSONL files produced by
``scripts/chunk_docs.py`` and consumed by ``scripts/build_index.py``.

File format: one JSON object per line, each being the ``dataclasses.asdict()``
representation of a :class:`~docuseek.chunking.base.Chunk`.  On deserialisation
``chunk_id`` is stripped and recomputed from content as an integrity check.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

import structlog

from docuseek.chunking.base import Chunk

logger = structlog.get_logger(__name__)

CHUNKS_DIR = Path("data/processed")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def chunks_jsonl_path(algorithm: str) -> Path:
    """Canonical output path for a chunking algorithm's JSONL file.

    Args:
        algorithm: Chunker algorithm name (e.g. ``"fixed"``, ``"semantic"``).

    Returns:
        Path like ``data/processed/chunks_semantic.jsonl``.
    """
    return CHUNKS_DIR / f"chunks_{algorithm}.jsonl"


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------


def chunk_to_json(chunk: Chunk) -> str:
    """Serialise a Chunk to a single JSON string (no trailing newline)."""
    return json.dumps(dataclasses.asdict(chunk), ensure_ascii=False, default=str)


def chunk_from_dict(data: dict[str, Any]) -> Chunk:
    """Deserialise a dict (from ``json.loads``) back into a Chunk.

    ``chunk_id`` is ``init=False`` on the dataclass, so it is stripped
    before construction and recomputed deterministically in
    ``__post_init__``.  This doubles as an integrity check — if content
    was edited, the recomputed ID reflects the change.

    Raises:
        TypeError: If required Chunk fields are missing from *data*.
    """
    return Chunk(**{k: v for k, v in data.items() if k != "chunk_id"})


# ---------------------------------------------------------------------------
# File I/O
# ---------------------------------------------------------------------------


def load_done_urls(path: Path) -> set[str]:
    """Stream through JSONL and return all completed doc URLs.

    All URLs except the last one seen are guaranteed complete.
    The last URL may be a partial crash — excluded so it gets re-chunked.
    Streams line by line: no Chunk objects ever built.
    """
    if not path.exists():
        return set()

    seen: list[str] = []  # ordered, to identify the last one
    seen_set: set[str] = set()

    with path.open(encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                url = json.loads(stripped)["doc_url"]
            except (json.JSONDecodeError, KeyError):
                continue
            if url not in seen_set:
                seen.append(url)
                seen_set.add(url)

    if not seen:
        return set()

    return seen_set - {seen[-1]}  # re-chunk the last doc to be safe


def load_chunks_jsonl(path: Path) -> list[Chunk]:
    """Load all chunks from a JSONL file.

    Reads line-by-line to avoid holding the entire file in memory twice.
    Skips blank lines and malformed entries with a warning rather than
    crashing, so that a partially-corrupted file is still usable.

    Returns:
        List of Chunk instances.  Empty if the file does not exist.
    """
    if not path.exists():
        return []

    chunks: list[Chunk] = []
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                chunks.append(chunk_from_dict(json.loads(stripped)))
            except (json.JSONDecodeError, TypeError) as exc:
                logger.warning("skipping_malformed_chunk_line", line=line_no, error=str(exc))

    logger.info("chunks_loaded", path=str(path), count=len(chunks))
    return chunks


def append_chunks_jsonl(path: Path, chunks: list[Chunk]) -> None:
    """Append chunks for a single document and flush to disk immediately.

    The explicit ``flush()`` ensures that if the process crashes after
    this call returns, the chunks are persisted — at most one document's
    worth of work is lost on any failure.
    """
    with path.open("a", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(chunk_to_json(chunk) + "\n")
        f.flush()
