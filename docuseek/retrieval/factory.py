"""
docuseek/retrieval/factory.py
-----------------------------
Factory function for constructing retriever instances from experiment config.

This is the only place in the library that imports ``ExperimentConfig``.
Individual retriever classes remain oblivious to the config schema — they
accept plain constructor arguments. The factory translates config language
into construction language.

Usage (in scripts only)::

    from docuseek.retrieval.factory import get_retriever
    from docuseek.experiment_config import ExperimentConfig

    config = ExperimentConfig.from_yaml(Path("experiments/00_baseline/config.yaml"))
    retriever = get_retriever(config.retriever)
"""

from __future__ import annotations

from docuseek.embedding.dense import DenseEmbedder
from docuseek.experiment_config import RetrieverConfig
from docuseek.retrieval.base import BaseRetriever
from docuseek.retrieval.bm25 import BM25Retriever
from docuseek.retrieval.dense import DenseRetriever
from docuseek.retrieval.hybrid import HybridRetriever


def get_retriever(config: RetrieverConfig) -> BaseRetriever:
    """Construct a retriever from a ``RetrieverConfig``.

    Args:
        config: Retriever section of the experiment config.

    Returns:
        A ready-to-use retriever instance.

    Raises:
        ValueError: If ``config.mode`` is not a known retiever.
    """
    match config.mode:
        case "sparse":
            return BM25Retriever()
        case "dense":
            return DenseRetriever(embedder=DenseEmbedder())
        case "hybrid":
            return HybridRetriever(
                dense=DenseRetriever(embedder=DenseEmbedder()),
                bm25=BM25Retriever(),
                rrf_k=config.rrf_k,
            )
        case _:
            raise ValueError(
                f"Unknown retriever mode: {config.mode!r}. Available: dense, hybrid, sparse."
            )
