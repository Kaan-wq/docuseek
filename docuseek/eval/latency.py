"""
docuseek/eval/latency.py
-------------------------
Pure functions for retrieval latency measurement and aggregation.

Encoding latency (query embedding) and index search latency (Qdrant query)
are kept separate so per-component costs can be reported independently.
This distinction is essential when comparing retrieval modes: dense and
hybrid share the same encoding step but differ in Qdrant search cost.

All latency values are in milliseconds throughout.

Design
------
Three layers:

- ``LatencySample``         — one data point: (encoding_ms, search_ms) per query.
- ``LatencyStats``          — aggregated statistics for a single component
                              (p50, p95, mean over N samples).
- ``RetrievalLatencyStats`` — combines encoding + search + total stats for a
                              full pipeline run.

Warmup logic is intentionally absent here — it is an orchestration concern
that belongs in ``scripts/benchmark.py``. This module is pure math only.

Percentile method
-----------------
Nearest-rank percentile on a sorted list: ``ceil(p / 100 * n) - 1``, clamped
to [0, n-1]. This is the same method used by numpy's default and avoids the
interpolation ambiguity of the ``statistics.quantiles`` API.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LatencySample:
    """Per-query latency breakdown for a single retrieval call.

    Attributes:
        encoding_ms: Time spent encoding the query into a vector (or
                     tokenising it for BM25). Identical for dense and
                     hybrid — separating it avoids attributing encoding
                     cost to Qdrant when comparing modes.
        search_ms:   Time spent inside Qdrant — from sending the query
                     to receiving ranked results. For hybrid this includes
                     RRF fusion inside Qdrant.
    """

    encoding_ms: float
    search_ms: float

    @property
    def total_ms(self) -> float:
        """Total wall-clock cost of one retrieval call."""
        return self.encoding_ms + self.search_ms


@dataclass(frozen=True)
class LatencyStats:
    """Aggregated latency statistics for a single pipeline component.

    Attributes:
        mean_ms: Arithmetic mean. Reported for completeness but not the
                 headline — a few slow outliers distort it.
        p50_ms:  Median. Typical user experience.
        p95_ms:  95th percentile. Worst realistic case engineers care about.
        n:       Number of samples this was computed from.
    """

    mean_ms: float
    p50_ms: float
    p95_ms: float
    n: int


@dataclass(frozen=True)
class RetrievalLatencyStats:
    """Full latency breakdown for one retrieval experiment.

    Attributes:
        encoding: Stats for the query encoding step only.
        search:   Stats for the Qdrant index search step only.
        total:    Stats for encoding + search combined.
    """

    encoding: LatencyStats
    search: LatencyStats
    total: LatencyStats


# ---------------------------------------------------------------------------
# Core math
# ---------------------------------------------------------------------------


def _percentile(sorted_samples: list[float], p: int) -> float:
    """Nearest-rank percentile from a pre-sorted list.

    Args:
        sorted_samples: Ascending-sorted list of floats. Must be non-empty.
        p:              Percentile in [1, 100].

    Returns:
        The value at the p-th percentile.
    """
    n = len(sorted_samples)
    idx = max(0, math.ceil(p / 100 * n) - 1)
    return sorted_samples[idx]


def compute_latency_stats(samples: list[float]) -> LatencyStats:
    """Compute p50, p95, and mean from a flat list of latency values.

    Args:
        samples: List of latency measurements in milliseconds. May be empty.

    Returns:
        ``LatencyStats`` with all fields zero and ``n=0`` when *samples*
        is empty, so callers never have to guard against missing data.
    """
    if not samples:
        return LatencyStats(mean_ms=0.0, p50_ms=0.0, p95_ms=0.0, n=0)

    sorted_samples = sorted(samples)
    return LatencyStats(
        mean_ms=statistics.mean(sorted_samples),
        p50_ms=_percentile(sorted_samples, 50),
        p95_ms=_percentile(sorted_samples, 95),
        n=len(sorted_samples),
    )


def aggregate_latency(samples: list[LatencySample]) -> RetrievalLatencyStats:
    """Aggregate per-query ``LatencySample``s into ``RetrievalLatencyStats``.

    Args:
        samples: One ``LatencySample`` per query, in any order.

    Returns:
        ``RetrievalLatencyStats`` with stats for encoding, search, and total.
        Returns all-zero stats when *samples* is empty.
    """
    return RetrievalLatencyStats(
        encoding=compute_latency_stats([s.encoding_ms for s in samples]),
        search=compute_latency_stats([s.search_ms for s in samples]),
        total=compute_latency_stats([s.total_ms for s in samples]),
    )
