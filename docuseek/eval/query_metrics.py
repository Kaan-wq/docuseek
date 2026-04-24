"""
docuseek/eval/query_metrics.py
--------------------------------
Pure functions for query rewriting cost and quality measurement.

Tracks three dimensions per rewriting method independently:
- Latency:    wall-clock time to produce variants (p50, p95, mean).
- Token cost: input and output tokens — a direct proxy for API cost when
              swapping local models for commercial ones (Claude, GPT).
              Tracked for all methods including NER, since GLiNER exposes
              a paid API and token counts future-proof the comparison.
- Diversity:  how lexically different variants are from the original query.
              A rewriter that produces near-identical variants adds latency
              and token cost with no retrieval benefit.

Design
------
Two layers:

- ``QueryMethodSample``  — one data point per (question, method): latency,
                           token counts, original query, and produced variants.
- ``QueryMethodStats``   — aggregated statistics across all questions for one
                           method: latency stats, mean token counts, mean
                           variant count, and mean diversity score.

Each method (NER, HyDE, multi-query) produces its own list of samples so
their costs can be compared in isolation. Aggregation is per-method.

Diversity metric
----------------
Mean pairwise Jaccard similarity between the original query and each variant,
subtracted from 1: diversity = 1 - mean_jaccard. Word-level tokenisation
(lowercase split) is used — sufficient for measuring lexical overlap between
short query strings without requiring a full tokenizer.
"""

from __future__ import annotations

from dataclasses import dataclass

from docuseek.eval.latency import LatencyStats, compute_latency_stats

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class QueryMethodSample:
    """Per-query cost sample for a single rewriting method.

    Attributes:
        method:        Rewriter name: ``"ner"``, ``"hyde"``, or ``"multi_query"``.
        latency_ms:    Wall-clock time to produce all variants, in milliseconds.
        input_tokens:  Tokens fed to the model. For NER: input query tokens.
                       For HyDE/multi-query: full prompt tokens.
        output_tokens: Tokens produced by the model. For NER: number of entity
                       annotations found. For HyDE/multi-query: generated text
                       tokens.
        original:      Original query string before any rewriting.
        variants:      Variants produced by this method, excluding the original.
                       Stored as a tuple so the dataclass remains hashable.
    """

    method: str
    latency_ms: float
    input_tokens: int
    output_tokens: int
    original: str
    variants: tuple[str, ...]


@dataclass(frozen=True)
class QueryMethodStats:
    """Aggregated cost statistics for one rewriting method across all questions.

    Attributes:
        method:             Rewriter name.
        latency:            p50/p95/mean latency statistics.
        mean_input_tokens:  Average input tokens per query.
        mean_output_tokens: Average output tokens per query.
        mean_n_variants:    Average number of variants produced per query.
        mean_diversity:     Average diversity score in [0, 1].
                            0 = all variants identical to original.
                            1 = no lexical overlap with original.
        n:                  Number of questions this was computed from.
    """

    method: str
    latency: LatencyStats
    mean_input_tokens: float
    mean_output_tokens: float
    mean_n_variants: float
    mean_diversity: float
    n: int


# ---------------------------------------------------------------------------
# Diversity math
# ---------------------------------------------------------------------------


def jaccard_similarity(text_a: str, text_b: str) -> float:
    """Jaccard similarity between two texts based on lowercased word sets.

    Args:
        text_a: First text.
        text_b: Second text.

    Returns:
        Float in [0, 1]. 1.0 = identical token sets. 0.0 = no shared tokens.
        Returns 1.0 when both texts are empty, 0.0 when only one is empty.
    """
    tokens_a = set(text_a.lower().split())
    tokens_b = set(text_b.lower().split())
    if not tokens_a and not tokens_b:
        return 1.0
    if not tokens_a or not tokens_b:
        return 0.0
    return len(tokens_a & tokens_b) / len(tokens_a | tokens_b)


def query_diversity(original: str, variants: list[str]) -> float:
    """Mean lexical diversity of variants relative to the original query.

    Diversity = 1 - mean Jaccard similarity between the original and each
    variant. High diversity means the rewriter is producing genuinely
    different phrasings worth the extra retrieval calls.

    Args:
        original: Original query string.
        variants: Rewritten variants, excluding the original.

    Returns:
        Float in [0, 1]. Returns 0.0 when variants is empty.
    """
    if not variants:
        return 0.0
    similarities = [jaccard_similarity(original, v) for v in variants]
    return 1.0 - (sum(similarities) / len(similarities))


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def aggregate_query_metrics(samples: list[QueryMethodSample]) -> QueryMethodStats:
    """Aggregate per-query samples into corpus-level stats for one method.

    Diversity is computed per question (using that question's original query
    and its variants), then averaged across questions.

    Args:
        samples: One ``QueryMethodSample`` per question, all from the same
                 method. Must not mix methods.

    Returns:
        ``QueryMethodStats`` with all fields zeroed and ``n=0`` when samples
        is empty, so callers never have to guard against missing data.
    """
    if not samples:
        zero = LatencyStats(mean_ms=0.0, p50_ms=0.0, p95_ms=0.0, n=0)
        return QueryMethodStats(
            method="unknown",
            latency=zero,
            mean_input_tokens=0.0,
            mean_output_tokens=0.0,
            mean_n_variants=0.0,
            mean_diversity=0.0,
            n=0,
        )

    n = len(samples)
    method = samples[0].method

    return QueryMethodStats(
        method=method,
        latency=compute_latency_stats([s.latency_ms for s in samples]),
        mean_input_tokens=sum(s.input_tokens for s in samples) / n,
        mean_output_tokens=sum(s.output_tokens for s in samples) / n,
        mean_n_variants=sum(len(s.variants) for s in samples) / n,
        mean_diversity=sum(query_diversity(s.original, list(s.variants)) for s in samples) / n,
        n=n,
    )
