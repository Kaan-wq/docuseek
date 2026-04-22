"""
docuseek/eval/benchmark.py
---------------------------
Reusable building blocks for the retrieval benchmark.

This module contains only pure utilities that are importable by other
eval modules (ragas_runner, llm_judge, future ablation scripts):

- ``load_gold_set``  — load and validate gold_set_v1.jsonl
- ``aggregate``      — average per-question scores into corpus-level metrics

Orchestration (building the retriever, looping over questions, printing
results, saving to disk) lives in ``scripts/benchmark.py``, which is the
single file to read to understand how a full benchmark run works.
"""

from __future__ import annotations

import json
from pathlib import Path

import structlog

from docuseek.eval.schema import GoldQuestion

logger = structlog.get_logger()

_DEFAULT_GOLD_SET = Path("data/eval/gold_set_v1.jsonl")


# ---------------------------------------------------------------------------
# Gold set loader
# ---------------------------------------------------------------------------


def load_gold_set(path: Path = _DEFAULT_GOLD_SET) -> list[GoldQuestion]:
    """Load and validate the gold evaluation set from a JSONL file.

    Validates every line against ``GoldQuestion`` so malformed entries
    surface immediately rather than causing silent metric errors mid-run.

    Args:
        path: Path to the gold set JSONL file.

    Returns:
        List of validated ``GoldQuestion`` objects, one per non-empty line.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValidationError:   If any line fails ``GoldQuestion`` validation.
    """
    questions: list[GoldQuestion] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line_stripped = line.strip()
            if line_stripped:
                questions.append(GoldQuestion(**json.loads(line_stripped)))
    logger.info("gold_set_loaded", path=str(path), count=len(questions))
    return questions


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def aggregate(scores: list[dict[str, float]]) -> dict[str, float]:
    """Average per-question metric scores into corpus-level metrics.

    Applies standard naming conventions for corpus-level metrics:
    - ``rr``   → ``mrr``   (Mean Reciprocal Rank)
    - ``ap@k`` → ``map@k`` (Mean Average Precision)

    All other keys are averaged as-is (e.g. ``ndcg@10``, ``recall@100``).

    Args:
        scores: List of per-question metric dicts, each produced by
                ``retrieval_metrics.compute_all()``.

    Returns:
        Dict of corpus-level metric scores. Empty dict if input is empty.
    """
    if not scores:
        return {}

    keys = scores[0].keys()
    result: dict[str, float] = {}

    for key in keys:
        mean_val = sum(s[key] for s in scores) / len(scores)
        if key == "rr":
            result["mrr"] = mean_val
        elif key.startswith("ap@"):
            result[key.replace("ap@", "map@")] = mean_val
        else:
            result[key] = mean_val

    return result
