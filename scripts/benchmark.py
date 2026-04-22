"""
scripts/benchmark.py
---------------------
CLI entry point for running the DocuSeek retrieval benchmark.

Loads an experiment config, builds the retriever, evaluates it against
the gold set, prints a Rich summary table, and saves results to disk.

Usage
-----
    uv run python scripts/benchmark.py --config experiments/00_baseline/config.yaml

Results are written to the same directory as the config::

    experiments/00_baseline/results.json

The full resolved config is archived inside results.json alongside the
metric numbers, so any run is fully reproducible from its output directory.

Metric strategy
---------------
Two k values are evaluated per question:

- ``k_primary`` (default 10): NDCG@k, Precision@k, AP@k, RR.
  These are the headline metrics used to compare experiments.

- ``k_recall`` (default 100): Recall@k only.
  Measures first-stage coverage independently of ranking quality.
  High Recall@100 with low NDCG@10 → retrieval is finding the right docs
  but ranking them poorly (reranker would help).
  Low Recall@100 → the corpus or chunking strategy is the bottleneck.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

import structlog
from rich.console import Console
from rich.table import Table

from docuseek.eval.benchmark import aggregate, load_gold_set
from docuseek.eval.retrieval_metrics import compute_all, recall_at_k
from docuseek.experiment_config import ExperimentConfig
from docuseek.logging import configure_logging
from docuseek.retrieval.factory import get_retriever

logger = structlog.get_logger()
console = Console()


# ---------------------------------------------------------------------------
# Core orchestration
# ---------------------------------------------------------------------------


def run_benchmark(config: ExperimentConfig) -> dict:
    """Run the full retrieval benchmark for a given experiment config.

    Loops over every question in the gold set, retrieves chunks, deduplicates
    retrieved URLs to doc-level (multiple chunks from the same page count as
    one hit at the rank of the first chunk), computes per-question metrics at
    both ``k_primary`` and ``k_recall``, then aggregates into corpus-level
    scores stratified by library and difficulty.

    Args:
        config: Fully validated experiment config loaded from YAML.

    Returns:
        Full results dict (aggregate + breakdowns). Also saved to disk.
    """
    questions = load_gold_set(Path(config.eval.gold_set_path))
    k_p = config.eval.k_primary
    k_r = config.eval.k_recall

    logger.info(
        "benchmark_start",
        experiment=config.name,
        questions=len(questions),
        k_primary=k_p,
        k_recall=k_r,
    )

    retriever = get_retriever(config.retriever)

    # ── Per-question scoring ─────────────────────────────────────────────────
    all_scores: list[dict[str, float]] = []
    per_library: dict[str, list[dict[str, float]]] = defaultdict(list)
    per_difficulty: dict[str, list[dict[str, float]]] = defaultdict(list)

    for question in questions:
        # Retrieve at the larger of the two k values so one call covers both
        chunks = retriever.retrieve(question.question, top_k=max(k_p, k_r))

        # Deduplicate to doc-level preserving rank order.
        # Chunks from same URL count as one hit at rank of first chunk.
        retrieved_urls = list(dict.fromkeys(chunk.doc_url for chunk in chunks))

        # Primary metrics at k_primary
        scores = compute_all(
            retrieved_urls=retrieved_urls,
            gold_urls=question.source_urls,
            k=k_p,
        )
        # Recall at k_recall (only metric that needs the wider candidate pool)
        scores[f"recall@{k_r}"] = recall_at_k(
            retrieved_urls=retrieved_urls,
            gold_urls=question.source_urls,
            k=k_r,
        )

        all_scores.append(scores)
        per_library[question.library].append(scores)
        per_difficulty[question.difficulty].append(scores)

    # ── Aggregate ────────────────────────────────────────────────────────────
    agg = aggregate(all_scores)
    by_library = {lib: aggregate(s) for lib, s in per_library.items()}
    by_difficulty = {diff: aggregate(s) for diff, s in per_difficulty.items()}

    # ── Display ──────────────────────────────────────────────────────────────
    _print_results(config.name, agg, by_library, by_difficulty, k_p, k_r)

    # ── Save ─────────────────────────────────────────────────────────────────
    results = {
        "experiment": config.name,
        "description": config.description,
        "timestamp": datetime.now(UTC).isoformat(),
        # Full config archived so this results.json is self-contained
        "config": config.model_dump(),
        "aggregate": agg,
        "by_library": by_library,
        "by_difficulty": by_difficulty,
    }
    _save_results(results, config)
    return results


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------


def _print_results(
    experiment_name: str,
    agg: dict[str, float],
    by_library: dict[str, dict[str, float]],
    by_difficulty: dict[str, dict[str, float]],
    k_p: int,
    k_r: int,
) -> None:
    """Print a Rich summary of benchmark results to stdout."""

    # ── Aggregate ────────────────────────────────────────────────────────────
    agg_table = Table(title=f"[bold]{experiment_name}[/bold] — aggregate metrics")
    agg_table.add_column("Metric", style="cyan")
    agg_table.add_column("Score", justify="right", style="bold white")

    # Print headline metric first, then the rest in a consistent order
    ordered_keys = [
        f"ndcg@{k_p}",
        f"recall@{k_r}",
        "mrr",
        f"map@{k_p}",
        f"precision@{k_p}",
        # Fallback: any remaining keys not listed above
        *[
            k
            for k in agg
            if k not in {f"ndcg@{k_p}", f"recall@{k_r}", "mrr", f"map@{k_p}", f"precision@{k_p}"}
        ],
    ]
    for key in ordered_keys:
        if key in agg:
            agg_table.add_row(key, f"{agg[key]:.4f}")
    console.print(agg_table)

    # ── By difficulty ────────────────────────────────────────────────────────
    diff_table = Table(title="Breakdown by difficulty")
    diff_table.add_column("Difficulty", style="cyan")
    diff_table.add_column(f"ndcg@{k_p}", justify="right")
    diff_table.add_column(f"recall@{k_r}", justify="right")
    diff_table.add_column("mrr", justify="right")

    for diff in ("easy", "medium", "hard"):
        if diff in by_difficulty:
            s = by_difficulty[diff]
            diff_table.add_row(
                diff,
                f"{s.get(f'ndcg@{k_p}', 0):.4f}",
                f"{s.get(f'recall@{k_r}', 0):.4f}",
                f"{s.get('mrr', 0):.4f}",
            )
    console.print(diff_table)

    # ── By library ───────────────────────────────────────────────────────────
    lib_table = Table(title="Breakdown by library")
    lib_table.add_column("Library", style="cyan")
    lib_table.add_column(f"ndcg@{k_p}", justify="right")
    lib_table.add_column(f"recall@{k_r}", justify="right")
    lib_table.add_column("mrr", justify="right")

    for lib in sorted(by_library):
        s = by_library[lib]
        lib_table.add_row(
            lib,
            f"{s.get(f'ndcg@{k_p}', 0):.4f}",
            f"{s.get(f'recall@{k_r}', 0):.4f}",
            f"{s.get('mrr', 0):.4f}",
        )
    console.print(lib_table)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def _save_results(results: dict, config: ExperimentConfig) -> None:
    """Save results.json next to the experiment config that produced it.

    The output directory is inferred from the config path stored on the
    config object so that results always land beside their config YAML.

    Args:
        results: Full results dict including aggregate, breakdowns, and
                 archived config.
        config:  The experiment config for this run.
    """
    # Derive output directory from the experiment name — matches the
    # convention that config.yaml lives at experiments/{name}/config.yaml
    output_dir = Path("experiments") / config.name
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "results.json"

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    logger.info("results_saved", path=str(output_path))
    console.print(f"\n[green]✓[/green] Results saved to [cyan]{output_path}[/cyan]")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the DocuSeek retrieval benchmark for an experiment.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Example:\n  uv run python scripts/benchmark.py --config experiments/00_baseline/config.yaml",
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        metavar="PATH",
        help="Path to the experiment config YAML.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    configure_logging(log_level="info")
    args = _parse_args()

    if not args.config.exists():
        console.print(f"[red]Config not found:[/red] {args.config}")
        sys.exit(1)

    config = ExperimentConfig.from_yaml(args.config)
    logger.info("config_loaded", experiment=config.name, config_path=str(args.config))

    try:
        run_benchmark(config)
    except Exception:
        logger.exception("benchmark_failed", experiment=config.name)
        sys.exit(1)
