"""
scripts/validate_gold_set.py
-----------------------------
Validate the gold evaluation set against the GoldQuestion schema.

Parses every line of data/eval/gold_set_v1.jsonl, reports schema
violations, and prints a distribution summary across libraries and
difficulties. Run this after every edit to the gold set.

Usage
-----
uv run python scripts/validate_gold_set.py

# Use a different gold set file
uv run python scripts/validate_gold_set.py --path data/eval/gold_set_v2.jsonl
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from pydantic import ValidationError
from rich.console import Console
from rich.table import Table

from docuseek.eval.schema import GoldQuestion

console = Console()

DEFAULT_PATH = Path("data/eval/gold_set_v1.jsonl")


def validate(path: Path) -> tuple[list[GoldQuestion], list[tuple[int, str]]]:
    """
    Load and validate every line of the gold set file.

    Args:
        path: Path to the JSONL gold set file.

    Returns:
        A tuple of (valid_questions, errors), where errors is a list
        of (line_number, error_message) for any line that failed parsing
        or schema validation.
    """
    valid: list[GoldQuestion] = []
    errors: list[tuple[int, str]] = []

    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                valid.append(GoldQuestion(**data))
            except json.JSONDecodeError as e:
                errors.append((line_no, f"invalid JSON: {e.msg}"))
            except ValidationError as e:
                errors.append((line_no, f"schema error: {e.errors()[0]['msg']}"))

    return valid, errors


def print_summary(questions: list[GoldQuestion]) -> None:
    """
    Print a distribution summary across libraries and difficulties.

    Args:
        questions: Validated questions to summarise.
    """
    table = Table(title=f"Gold set summary — {len(questions)} questions")
    table.add_column("Library", style="cyan")
    table.add_column("Easy", justify="right")
    table.add_column("Medium", justify="right")
    table.add_column("Hard", justify="right")
    table.add_column("Total", justify="right", style="bold")

    counts: dict[str, Counter] = {}
    for q in questions:
        counts.setdefault(q.library, Counter())[q.difficulty] += 1

    for library in sorted(counts.keys()):
        c = counts[library]
        total = c["easy"] + c["medium"] + c["hard"]
        table.add_row(library, str(c["easy"]), str(c["medium"]), str(c["hard"]), str(total))

    total_easy = sum(c["easy"] for c in counts.values())
    total_medium = sum(c["medium"] for c in counts.values())
    total_hard = sum(c["hard"] for c in counts.values())
    table.add_row(
        "[bold]TOTAL[/bold]",
        f"[bold]{total_easy}[/bold]",
        f"[bold]{total_medium}[/bold]",
        f"[bold]{total_hard}[/bold]",
        f"[bold]{len(questions)}[/bold]",
    )

    console.print(table)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate the DocuSeek gold eval set.")
    parser.add_argument(
        "--path",
        type=Path,
        default=DEFAULT_PATH,
        help=f"Path to gold set JSONL file. Default: {DEFAULT_PATH}",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.path.exists():
        console.print(f"[red]File not found: {args.path}[/red]")
        sys.exit(1)

    console.rule(f"[bold green]Validating {args.path}")
    valid, errors = validate(args.path)

    if errors:
        console.print(f"[red]Found {len(errors)} errors:[/red]")
        for line_no, msg in errors:
            console.print(f"  line {line_no}: {msg}")
        console.print()

    if valid:
        print_summary(valid)

    if errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
