"""
docuseek/chunking/factory.py
-----------------------------
Factory function for constructing chunker instances from experiment config.

This is the only place in the library that imports ``ExperimentConfig``.
Individual chunker classes remain oblivious to the config schema — they
accept plain constructor arguments. The factory translates config language
into construction language.

Usage (in scripts only)::

    from docuseek.chunking.factory import get_chunker
    from docuseek.experiment_config import ExperimentConfig

    config = ExperimentConfig.from_yaml(Path("experiments/00_baseline/config.yaml"))
    chunker = get_chunker(config.chunker)
"""

from __future__ import annotations

from docuseek.chunking.base import BaseChunker
from docuseek.chunking.document_structure import MarkdownHeaderChunker
from docuseek.chunking.fixed import FixedSizeChunker
from docuseek.chunking.recursive import RecursiveChunker
from docuseek.experiment_config import ChunkerConfig


def get_chunker(config: ChunkerConfig) -> BaseChunker:
    """Construct a chunker from a ``ChunkerConfig``.

    Args:
        config: Chunker section of the experiment config.

    Returns:
        A ready-to-use chunker instance.

    Raises:
        ValueError: If ``config.algorithm`` is not a known chunker.
    """
    match config.algorithm:
        case "fixed":
            return FixedSizeChunker(
                chunk_size=config.chunk_size,
                overlap=config.chunk_overlap,
            )
        case "recursive":
            return RecursiveChunker(
                chunk_size=config.chunk_size,
                overlap=config.chunk_overlap,
            )
        case "markdown":
            return MarkdownHeaderChunker(
                chunk_size=config.chunk_size,
                overlap=config.chunk_overlap,
            )
        case "semantic":
            raise NotImplementedError(
                "Semantic chunker is not yet implemented. "
                "Available algorithms: fixed, recursive, markdown."
            )
        case "agentic":
            raise NotImplementedError(
                "Agentic chunker is not yet implemented. "
                "Available algorithms: fixed, recursive, markdown."
            )
        case _:
            raise ValueError(
                f"Unknown chunker algorithm: {config.algorithm!r}. "
                f"Available: fixed, recursive, markdown, semantic."
            )
