"""
scripts/chunk_docs.py
---------------------
Decoupled chunking stage: load clean docs → chunk → write to JSONL.

Produces ``data/processed/chunks_{algorithm}.jsonl``, one JSON object per
line, each representing a single Chunk.  This file is the input for
``build_index.py``, which handles embedding and Qdrant upsert separately.

Crash-safe resume
-----------------
Semantic chunking (and any future LLM-based chunker) is slow and
resource-intensive.  This script is designed to survive crashes:

1. On startup, if the output file already exists, load it and identify
   which documents were **fully** chunked (chunk count == chunk_total).
2. Discard chunks from partially-chunked documents (crash mid-doc).
3. Rewrite the file with only the complete documents' chunks.
4. Process remaining documents, flushing to disk after each one.

A crash mid-run loses at most one document's worth of work.

Usage
-----
    # Chunk using a specific experiment config
    uv run python scripts/chunk_docs.py --config experiments/01_chunking/semantic/config.yaml

    # Force re-chunking even if the output file already exists
    uv run python scripts/chunk_docs.py --config experiments/01_chunking/semantic/config.yaml --force
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import structlog
from rich.console import Console
from rich.progress import track

from docuseek.chunking.base import Chunk
from docuseek.chunking.factory import get_chunker
from docuseek.chunking.io import (
    append_chunks_jsonl,
    chunks_jsonl_path,
    load_chunks_jsonl,
    write_chunks_jsonl,
)
from docuseek.experiment_config import ExperimentConfig
from docuseek.ingestion.pipeline import load_jsonl
from docuseek.logging import configure_logging

logger = structlog.get_logger(__name__)
console = Console()

CLEAN_DOCS_PATH = Path("data/processed/huggingface.jsonl")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _filter_complete_docs(chunks: list[Chunk]) -> tuple[list[Chunk], set[str]]:
    """Keep only chunks from documents that were fully chunked.

    A document is considered complete when the number of chunks we hold
    for its ``doc_url`` equals the ``chunk_total`` declared on those
    chunks.  Partial documents (from a previous crash mid-doc) are
    discarded so they can be re-processed cleanly on the next run.

    Args:
        chunks: All chunks loaded from an existing JSONL file.

    Returns:
        Tuple of (chunks from complete documents, their ``doc_url`` values).
    """
    by_url: dict[str, list[Chunk]] = defaultdict(list)
    for chunk in chunks:
        by_url[chunk.doc_url].append(chunk)

    complete_chunks: list[Chunk] = []
    complete_urls: set[str] = set()

    for url, doc_chunks in by_url.items():
        expected = doc_chunks[0].chunk_total
        if len(doc_chunks) == expected:
            complete_chunks.extend(doc_chunks)
            complete_urls.add(url)
        else:
            logger.info(
                "discarding_incomplete_doc",
                doc_url=url,
                have=len(doc_chunks),
                expected=expected,
            )

    return complete_chunks, complete_urls


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def chunk_docs(config: ExperimentConfig, force: bool = False) -> None:
    """Run the chunking pipeline with crash-safe resume.

    Loads clean documents, chunks them using the algorithm specified in
    ``config.chunker``, and writes the results incrementally to a JSONL
    file.  Already-completed documents are skipped on re-runs.

    Args:
        config: Experiment configuration — chunker algo and params
                are read from ``config.chunker``.
        force:  If True, discard any existing chunk file and start fresh.
    """
    algorithm = config.chunker.algorithm
    output = chunks_jsonl_path(algorithm)

    # ── 1. Handle resume vs fresh start ───────────────────────────────────
    done_urls: set[str] = set()

    if force and output.exists():
        output.unlink()
        logger.info("forced_restart", deleted=str(output))

    if output.exists():
        existing = load_chunks_jsonl(output)
        complete, done_urls = _filter_complete_docs(existing)

        # Rewrite file with only complete docs (discard partial crash debris)
        if len(complete) < len(existing):
            write_chunks_jsonl(output, complete)
            logger.info(
                "cleaned_partial_chunks",
                kept=len(complete),
                discarded=len(existing) - len(complete),
            )

        logger.info("resuming", complete_docs=len(done_urls), chunks_on_disk=len(complete))
    else:
        output.parent.mkdir(parents=True, exist_ok=True)

    # ── 2. Load clean docs and filter already-done ────────────────────────
    docs = load_jsonl(CLEAN_DOCS_PATH)
    remaining = [doc for doc in docs if doc.url not in done_urls]

    logger.info(
        "docs_loaded",
        total=len(docs),
        already_chunked=len(docs) - len(remaining),
        to_process=len(remaining),
    )

    if not remaining:
        console.print("[green]All documents already chunked — nothing to do.[/green]")
        return

    # ── 3. Chunk and write incrementally ──────────────────────────────────
    chunker = get_chunker(config.chunker)
    total_chunks = 0
    errors = 0

    for doc in track(remaining, description=f"Chunking ({algorithm})"):
        try:
            doc_chunks = chunker.chunk(doc)
            append_chunks_jsonl(output, doc_chunks)
            total_chunks += len(doc_chunks)
        except Exception:
            errors += 1
            logger.exception("chunking_failed", doc_url=doc.url)
            # Skip this doc and continue — it will be retried on next run
            # since its URL never enters the complete set.
            continue

    logger.info(
        "chunking_complete",
        algorithm=algorithm,
        new_chunks=total_chunks,
        errors=errors,
        output=str(output),
    )

    if errors:
        console.print(f"[yellow]{errors} document(s) failed — re-run to retry them.[/yellow]")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Chunk clean documents into a JSONL file for indexing.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        metavar="PATH",
        help="Path to the experiment config YAML.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="Delete existing chunk file and start from scratch.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    configure_logging(log_level="info")
    args = parse_args()
    try:
        config = ExperimentConfig.from_yaml(args.config)
        chunk_docs(config=config, force=args.force)
    except Exception:
        logger.exception("chunk_docs_failed")
        sys.exit(1)
