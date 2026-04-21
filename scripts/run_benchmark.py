"""
scripts/run_benchmark.py
-------------------------
CLI entry point for running the DocuSeek retrieval benchmark.

Usage
-----
uv run python scripts/run_benchmark.py --experiment baseline_fixed
uv run python scripts/run_benchmark.py --experiment baseline_recursive --k 5
"""

import argparse
import sys

import structlog

from docuseek.embedding.dense import DenseEmbedder
from docuseek.eval.benchmark import run_benchmark
from docuseek.logging import configure_logging
from docuseek.retrieval.dense import DenseRetriever

logger = structlog.get_logger()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the DocuSeek retrieval benchmark.")
    parser.add_argument("--experiment", required=True, help="Experiment name, e.g. baseline_fixed.")
    parser.add_argument("--k", type=int, default=10, help="Retrieval cutoff rank. Default: 10.")
    return parser.parse_args()


if __name__ == "__main__":
    configure_logging(log_level="info")
    args = parse_args()
    embedder = DenseEmbedder()
    retriever = DenseRetriever(embedder=embedder)
    try:
        run_benchmark(retriever=retriever, experiment_name=args.experiment, k=args.k)
    except Exception:
        logger.exception("benchmark_failed")
        sys.exit(1)
