"""
docuseek/eval/benchmark.py
----------------------------
Runs the full retrieval benchmark against the gold evaluation set.

Loads gold_set_v1.jsonl, retrieves chunks for each question using the
configured retriever, computes per-question retrieval metrics, and
aggregates them into corpus-level scores (MAP@k, MRR, mean NDCG@k, etc).

Results are saved to experiments/{experiment_name}/results.json so every
run is reproducible and comparable across experiments.
"""

import json
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

import structlog
from rich.console import Console
from rich.table import Table

from docuseek.eval.retrieval_metrics import compute_all
from docuseek.eval.schema import GoldQuestion
from docuseek.retrieval.dense import DenseRetriever

logger = structlog.get_logger()
console = Console()

GOLD_SET_PATH = Path("data/eval/gold_set_v1.jsonl")


def load_gold_set(path: Path = GOLD_SET_PATH) -> list[GoldQuestion]:
    """
    Load and validate the gold evaluation set from a JSONL file.

    Args:
        path: Path to the gold set JSONL file.

    Returns:
        List of validated GoldQuestion objects.
    """
    questions = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line_ = line.strip()
            if line_:
                questions.append(GoldQuestion(**json.loads(line_)))
    return questions


def run_benchmark(
    retriever: DenseRetriever,
    experiment_name: str,
    k: int = 10,
    gold_set_path: Path = GOLD_SET_PATH,
) -> dict[str, float]:
    """
    Run the full retrieval benchmark against the gold set.

    For each question in the gold set, retrieves the top-k chunks,
    extracts their source URLs, and computes retrieval metrics against
    the gold source URLs. Aggregates per-question scores into corpus-level
    means and saves results to disk.

    Args:
        retriever:       Configured retriever instance to benchmark.
        experiment_name: Identifier for this run, e.g. "fixed_chunker_dense".
                         Results saved to experiments/{name}/results.json.
        k:               Retrieval cutoff rank for all @k metrics.
        gold_set_path:   Path to the gold set JSONL file.

    Returns:
        Dict of aggregated corpus-level metric scores.
    """
    questions = load_gold_set(gold_set_path)
    logger.info("benchmark_start", questions=len(questions), k=k, experiment=experiment_name)

    # ── Per-question scores ───────────────────────────────────────────────────
    all_scores: list[dict[str, float]] = []
    per_library: dict[str, list[dict[str, float]]] = defaultdict(list)
    per_difficulty: dict[str, list[dict[str, float]]] = defaultdict(list)

    for question in questions:
        chunks = retriever.retrieve(question.question, top_k=k)
        retrieved_urls = [chunk.doc_url for chunk in chunks]

        scores = compute_all(
            retrieved_urls=retrieved_urls,
            gold_urls=question.source_urls,
            k=k,
        )
        all_scores.append(scores)
        per_library[question.library].append(scores)
        per_difficulty[question.difficulty].append(scores)

    # ── Aggregate ─────────────────────────────────────────────────────────────
    aggregate = _aggregate(all_scores, prefix="")
    by_library = {lib: _aggregate(scores) for lib, scores in per_library.items()}
    by_difficulty = {diff: _aggregate(scores) for diff, scores in per_difficulty.items()}

    # ── Print summary table ───────────────────────────────────────────────────
    _print_results(aggregate, by_library, by_difficulty, k)

    # ── Save to disk ──────────────────────────────────────────────────────────
    results = {
        "experiment": experiment_name,
        "timestamp": datetime.now(UTC).isoformat(),
        "k": k,
        "aggregate": aggregate,
        "by_library": by_library,
        "by_difficulty": by_difficulty,
    }
    _save_results(results, experiment_name)
    return aggregate


def _aggregate(scores: list[dict[str, float]]) -> dict[str, float]:
    """
    Average per-question scores into corpus-level metrics.

    rr  → mrr  (Mean Reciprocal Rank)
    ap@k → map@k (Mean Average Precision)
    All others prefixed with mean_.

    Args:
        scores: List of per-question metric dicts from compute_all().

    Returns:
        Dict of aggregated metric scores.
    """
    if not scores:
        return {}

    keys = scores[0].keys()
    aggregated = {}
    for key in keys:
        mean_val = sum(s[key] for s in scores) / len(scores)
        # Rename to standard corpus-level metric names
        if key == "rr":
            aggregated["mrr"] = mean_val
        elif key.startswith("ap@"):
            aggregated[key.replace("ap@", "map@")] = mean_val
        else:
            aggregated[key] = mean_val
    return aggregated


def _print_results(
    aggregate: dict[str, float],
    by_library: dict[str, dict[str, float]],
    by_difficulty: dict[str, dict[str, float]],
    k: int,
) -> None:
    """Print a Rich summary table of benchmark results."""
    # ── Aggregate table ───────────────────────────────────────────────────────
    table = Table(title=f"Retrieval benchmark — aggregate (k={k})")
    table.add_column("Metric", style="cyan")
    table.add_column("Score", justify="right", style="bold")
    for metric, score in aggregate.items():
        table.add_row(metric, f"{score:.4f}")
    console.print(table)

    # ── By difficulty ─────────────────────────────────────────────────────────
    diff_table = Table(title="NDCG@K by difficulty")
    diff_table.add_column("Difficulty", style="cyan")
    diff_table.add_column(f"ndcg@{k}", justify="right")
    diff_table.add_column(f"recall@{k}", justify="right")
    diff_table.add_column("mrr", justify="right")
    for diff in ("easy", "medium", "hard"):
        if diff in by_difficulty:
            s = by_difficulty[diff]
            diff_table.add_row(
                diff,
                f"{s.get(f'ndcg@{k}', 0):.4f}",
                f"{s.get(f'recall@{k}', 0):.4f}",
                f"{s.get('mrr', 0):.4f}",
            )
    console.print(diff_table)


def _save_results(results: dict, experiment_name: str) -> None:
    """
    Save benchmark results to experiments/{experiment_name}/results.json.

    Args:
        results:         Full results dict including aggregate and breakdowns.
        experiment_name: Used as the subdirectory name under experiments/.
    """
    output_dir = Path("experiments") / experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "results.json"
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    logger.info("results_saved", path=str(output_path))
