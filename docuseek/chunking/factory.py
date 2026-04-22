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

from docuseek.chunking.agentic import AgenticChunker
from docuseek.chunking.base import BaseChunker
from docuseek.chunking.context_aware import ContextAwareChunker
from docuseek.chunking.document_structure import MarkdownHeaderChunker
from docuseek.chunking.fixed import FixedSizeChunker
from docuseek.chunking.recursive import RecursiveChunker
from docuseek.chunking.semantic import SemanticChunker
from docuseek.experiment_config import ChunkerConfig


def get_chunker(config: ChunkerConfig) -> BaseChunker:
    """Wrapper around ``_build_chunker`` for ``ContextAwareChunker``"""
    chunker = _build_chunker(config)
    if config.context_aware:
        return ContextAwareChunker(inner=chunker)
    return chunker


def _build_chunker(config: ChunkerConfig) -> BaseChunker:
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
            return SemanticChunker(
                embedder=None,  # from settings
                threshold=None,  # from experiment_config
                min_chunk_size=None,  # from experiment_config
                max_chunk_size=config.chunk_size,
            )
        case "agentic":
            return AgenticChunker(
                window_size=None,  # from experiment_config
                max_chunk_size=config.chunk_size,
            )
        case _:
            raise ValueError(
                f"Unknown chunker algorithm: {config.algorithm!r}. "
                f"Available: agentic, fixed, markdown, recursive, semantic."
            )
