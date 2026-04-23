"""
docuseek/experiment_config.py
------------------------------
Pydantic model for per-experiment algorithmic configuration.

This file is the single source of truth for every choice that varies
between experiments: chunking algorithm, retrieval mode, reranker, query
strategies, and generation techniques.

Separation of concerns
-----------------------
- ``docuseek/config.py`` / ``.env``  — infrastructure, secrets, and fixed model
  assets (API keys, endpoints, model names, embedding dimensions). These values
  are the same for every experiment.
- ``docuseek/experiment_config.py``  — algorithmic choices that differ between
  experiments. Nothing here should be a model name or a secret.

This distinction enforces fair experimental comparisons: when two experiments
differ only in the value of one field here, any change in metric is attributable
to that field alone, not to a different model or infrastructure choice.

Experiment layout
-----------------
Each experiment lives in its own directory::

    experiments/
        00_baseline/
            config.yaml      ← validated against ExperimentConfig
            results.json     ← written by benchmark.py, includes archived config

Loading a config::

    config = ExperimentConfig.from_yaml(Path("experiments/00_baseline/config.yaml"))

Design notes
------------
- Algorithm selection fields use ``Literal`` — exactly one option is chosen.
- Optional technique fields are independent booleans — they compose freely.
  For example, ``context_aware`` can be combined with any chunking algorithm;
  ``cot`` and ``few_shot`` can both be enabled simultaneously.
- ``build_index.py`` and ``benchmark.py`` both consume this config. The index
  must be built with the same chunker config the benchmark declares, otherwise
  retrieval metrics are meaningless.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Sub-configs
# ---------------------------------------------------------------------------


class ChunkerConfig(BaseModel):
    """Configuration for the chunking stage.

    ``algorithm``     selects the chunking strategy (pick exactly one).
    ``context_aware`` prepends surrounding document context to each chunk before embedding.
    """

    algorithm: Literal["agentic", "fixed", "markdown", "recursive", "semantic"]
    chunk_size: int = 500
    chunk_overlap: int = 50
    context_aware: bool = False
    threshold: float = 0.75
    min_chunk_size: int = 100
    window_size: int = 5


class RetrieverConfig(BaseModel):
    """Configuration for the retrieval stage.

    ``rrf_k``            is the RRF constant; only used when ``mode`` is ``hybrid``.
    """

    mode: Literal["dense", "sparse", "hybrid"]
    rrf_k: int = 60


class RerankerConfig(BaseModel):
    """Configuration for the optional reranking stage.

    The model checkpoint for each method is fixed in ``config.py``.
    This config only controls whether reranking is active and which
    technique is used — not which weights are loaded.
    """

    enabled: bool = False
    method: Literal["colbert", "cross_encoder"] = "colbert"


class QueryConfig(BaseModel):
    """Optional query transformation strategies.

    All flags are independent and can be combined freely.
    When multiple strategies are enabled, they are applied in order:
    NER → HyDE → multi-query.
    """

    ner: bool = False
    hyde: bool = False
    multi_query: bool = False


class GenerationConfig(BaseModel):
    """Prompt engineering techniques applied at generation time.

    All flags are independent and composable — ``cot`` and ``few_shot`` can
    both be enabled simultaneously (few-shot CoT is a standard technique).
    The generator model is fixed in ``config.py``.
    """

    cot: bool = False
    few_shot: bool = False
    budget_forcing: bool = False


class EvalConfig(BaseModel):
    """Evaluation settings."""

    gold_set_path: str = "data/eval/gold_set_v1.jsonl"
    # k_primary: used for NDCG@k and Precision@k (headline metric)
    k_primary: int = 10
    # k_recall: used for Recall@k (measures first-stage coverage)
    k_recall: int = 100


# ---------------------------------------------------------------------------
# Top-level config
# ---------------------------------------------------------------------------


class ExperimentConfig(BaseModel):
    """Full configuration for a single experiment run.

    Instantiate via ``from_yaml`` rather than directly:

        config = ExperimentConfig.from_yaml(Path("experiments/00_baseline/config.yaml"))
    """

    name: str
    description: str = ""
    chunker: ChunkerConfig
    retriever: RetrieverConfig
    reranker: RerankerConfig = Field(default_factory=RerankerConfig)
    query: QueryConfig = Field(default_factory=QueryConfig)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    eval: EvalConfig = Field(default_factory=EvalConfig)

    @classmethod
    def from_yaml(cls, path: Path) -> ExperimentConfig:
        """Load and validate an experiment config from a YAML file.

        Args:
            path: Path to the experiment ``config.yaml``.

        Returns:
            Validated ``ExperimentConfig`` instance.

        Raises:
            ValidationError: If the YAML does not match the schema.
            FileNotFoundError: If ``path`` does not exist.
        """
        return cls.model_validate(yaml.safe_load(path.read_text()))

    def save(self, path: Path) -> None:
        """Serialise this config to YAML.

        Intended to archive the resolved config alongside ``results.json``
        so that any run is fully reproducible from its output directory.

        Args:
            path: Destination path, e.g.
                ``experiments/00_baseline/config_resolved.yaml``.
        """
        path.write_text(yaml.dump(self.model_dump(), default_flow_style=False))
