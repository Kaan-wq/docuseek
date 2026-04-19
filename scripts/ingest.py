"""
scripts/ingest.py
-----------------
CLI entry point for the DocuSeek ingestion pipeline.

Usage
-----
# Run all scrapers (loads from disk if raw data exists)
uv run python scripts/ingest.py

# Force a fresh scrape even if raw data exists
uv run python scripts/ingest.py --force

# Run a specific scraper only
uv run python scripts/ingest.py --sources huggingface

# Combine both flags
uv run python scripts/ingest.py --sources huggingface --force
"""

import argparse
import sys
import time
from collections import Counter

import structlog
from rich.console import Console
from rich.table import Table

from docuseek.ingestion.pipeline import SCRAPERS, run_ingestion
from docuseek.logging import configure_logging

logger = structlog.get_logger()
console = Console()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the DocuSeek ingestion pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--sources",
        nargs="+",
        choices=list(SCRAPERS.keys()),
        default=None,
        metavar="SOURCE",
        help=(f"Which sources to scrape. Choices: {', '.join(SCRAPERS.keys())}. Defaults to all."),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="Re-scrape even if raw data already exists on disk.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    console.rule("[bold green]DocuSeek — Ingestion pipeline")
    console.print(f"  Sources  : {args.sources or list(SCRAPERS.keys())}")
    console.print(f"  Force    : {args.force}")
    console.print()

    start = time.perf_counter()

    try:
        docs = run_ingestion(
            scraper_names=args.sources,
            force_rescrape=args.force,
        )
    except KeyboardInterrupt:
        console.print(
            "\n[yellow]Interrupted by user — partial results may have been saved.[/yellow]"
        )
        sys.exit(1)
    except Exception:
        logger.exception("ingestion_failed")
        console.print("[bold red]Ingestion failed — see logs above.[/bold red]")
        sys.exit(1)

    elapsed = time.perf_counter() - start

    # ── summary table ────────────────────────────────────────────────────────
    table = Table(title="Ingestion summary", show_lines=True)
    table.add_column("Source", style="cyan")
    table.add_column("Clean docs", justify="right", style="green")

    counts: Counter[str] = Counter(doc.source for doc in docs)
    for source, count in sorted(counts.items()):
        table.add_row(source, str(count))
    table.add_row("[bold]TOTAL[/bold]", f"[bold]{len(docs)}[/bold]")

    console.print(table)
    console.print(f"\n✓ Done in [bold]{elapsed:.1f}s[/bold]")


if __name__ == "__main__":
    configure_logging(log_level="info")
    main()
