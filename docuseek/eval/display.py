"""
docuseek/eval/display.py
-------------------------
Rich display functions for benchmark results.

Separates rendering logic from orchestration so scripts/benchmark.py
stays focused on pipeline coordination, and so future scripts (result
comparisons, ablation viewers) can reuse these tables without importing
from scripts/.

Each private function renders one table. ``print_results`` composes them.
"""

from __future__ import annotations

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table

from docuseek.eval.latency import LatencyStats, RetrievalLatencyStats
from docuseek.eval.query_metrics import QueryMethodStats

console = Console()


# ---------------------------------------------------------------------------
# Private table renderers
# ---------------------------------------------------------------------------


def _print_aggregate_table(
    experiment_name: str,
    agg: dict[str, float],
    k_p: int,
    k_r: int,
) -> None:
    console.print(Rule(f"[bold cyan]{experiment_name}[/bold cyan]", style="cyan"))

    table = Table(box=box.ROUNDED, show_header=True, header_style="bold cyan")
    table.add_column("Metric", style="cyan")
    table.add_column("Score", justify="right", style="bold white")

    ordered_keys = [
        f"ndcg@{k_p}",
        f"recall@{k_r}",
        "mrr",
        f"map@{k_p}",
        f"precision@{k_p}",
        *[
            k
            for k in agg
            if k not in {f"ndcg@{k_p}", f"recall@{k_r}", "mrr", f"map@{k_p}", f"precision@{k_p}"}
        ],
    ]
    for key in ordered_keys:
        if key in agg:
            table.add_row(key, f"{agg[key]:.4f}")

    console.print(Panel(table, title="Quality metrics", border_style="cyan"))


def _print_difficulty_table(
    by_difficulty: dict[str, dict[str, float]],
    k_p: int,
    k_r: int,
) -> None:
    table = Table(box=box.ROUNDED, show_header=True, header_style="bold cyan")
    table.add_column("Difficulty", style="cyan")
    table.add_column(f"ndcg@{k_p}", justify="right")
    table.add_column(f"recall@{k_r}", justify="right")
    table.add_column("mrr", justify="right")

    for diff in ("easy", "medium", "hard"):
        if diff in by_difficulty:
            s = by_difficulty[diff]
            table.add_row(
                diff,
                f"{s.get(f'ndcg@{k_p}', 0):.4f}",
                f"{s.get(f'recall@{k_r}', 0):.4f}",
                f"{s.get('mrr', 0):.4f}",
            )

    console.print(Panel(table, title="By difficulty", border_style="cyan"))


def _print_library_table(
    by_library: dict[str, dict[str, float]],
    k_p: int,
    k_r: int,
) -> None:
    table = Table(box=box.ROUNDED, show_header=True, header_style="bold cyan")
    table.add_column("Library", style="cyan")
    table.add_column(f"ndcg@{k_p}", justify="right")
    table.add_column(f"recall@{k_r}", justify="right")
    table.add_column("mrr", justify="right")

    for lib in sorted(by_library):
        s = by_library[lib]
        table.add_row(
            lib,
            f"{s.get(f'ndcg@{k_p}', 0):.4f}",
            f"{s.get(f'recall@{k_r}', 0):.4f}",
            f"{s.get('mrr', 0):.4f}",
        )

    console.print(Panel(table, title="By library", border_style="cyan"))


def _print_latency_table(latency_stats: RetrievalLatencyStats) -> None:
    table = Table(box=box.ROUNDED, show_header=True, header_style="bold yellow")
    table.add_column("Component", style="yellow")
    table.add_column("p50 (ms)", justify="right")
    table.add_column("p95 (ms)", justify="right")
    table.add_column("mean (ms)", justify="right")

    for label, stats in [
        ("encoding", latency_stats.encoding),
        ("search", latency_stats.search),
        ("total", latency_stats.total),
    ]:
        table.add_row(
            label,
            f"{stats.p50_ms:.1f}",
            f"{stats.p95_ms:.1f}",
            f"{stats.mean_ms:.1f}",
        )

    console.print(Panel(table, title="Retrieval latency", border_style="yellow"))


def _print_reranker_table(reranker_stats: LatencyStats) -> None:
    table = Table(box=box.ROUNDED, show_header=True, header_style="bold yellow")
    table.add_column("Component", style="yellow")
    table.add_column("p50 (ms)", justify="right")
    table.add_column("p95 (ms)", justify="right")
    table.add_column("mean (ms)", justify="right")

    table.add_row(
        "reranker",
        f"{reranker_stats.p50_ms:.1f}",
        f"{reranker_stats.p95_ms:.1f}",
        f"{reranker_stats.mean_ms:.1f}",
    )

    console.print(Panel(table, title="Reranker latency", border_style="yellow"))


def _print_query_tables(query_stats: dict[str, QueryMethodStats]) -> None:
    for method, stats in query_stats.items():
        table = Table(box=box.ROUNDED, show_header=True, header_style="bold magenta")
        table.add_column("Metric", style="magenta")
        table.add_column("Value", justify="right")

        table.add_row("latency p50 (ms)", f"{stats.latency.p50_ms:.1f}")
        table.add_row("latency p95 (ms)", f"{stats.latency.p95_ms:.1f}")
        table.add_row("latency mean (ms)", f"{stats.latency.mean_ms:.1f}")
        table.add_row("mean input tokens", f"{stats.mean_input_tokens:.1f}")
        table.add_row("mean output tokens", f"{stats.mean_output_tokens:.1f}")
        table.add_row("mean variants", f"{stats.mean_n_variants:.1f}")
        table.add_row("mean diversity", f"{stats.mean_diversity:.3f}")

        console.print(Panel(table, title=f"Query rewriting — {method}", border_style="magenta"))


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------


def print_results(
    experiment_name: str,
    agg: dict[str, float],
    by_library: dict[str, dict[str, float]],
    by_difficulty: dict[str, dict[str, float]],
    latency_stats: RetrievalLatencyStats,
    reranker_stats: LatencyStats | None,
    query_stats: dict[str, QueryMethodStats],
    k_p: int,
    k_r: int,
) -> None:
    """Print a full Rich summary of benchmark results to stdout.

    Args:
        experiment_name: Name of the experiment, used as table title.
        agg:             Corpus-level aggregate metric scores.
        by_library:      Aggregate scores stratified by library.
        by_difficulty:   Aggregate scores stratified by difficulty.
        latency_stats:   Retrieval latency breakdown (encoding + search).
        reranker_stats:  Reranker latency stats, or None if disabled.
        query_stats:     Per-method query rewriting cost stats.
        k_p:             Primary k value used for NDCG/Precision/MAP.
        k_r:             Recall k value.
    """
    _print_aggregate_table(experiment_name, agg, k_p, k_r)
    console.print(Rule(style="cyan dim"))
    _print_difficulty_table(by_difficulty, k_p, k_r)
    _print_library_table(by_library, k_p, k_r)
    console.print(Rule(style="yellow dim"))
    _print_latency_table(latency_stats)
    if reranker_stats:
        _print_reranker_table(reranker_stats)
    if query_stats:
        console.print(Rule(style="magenta dim"))
        _print_query_tables(query_stats)
