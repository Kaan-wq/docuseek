"""
docuseek/reranking/factory.py
------------------------------
Factory for instantiating rerankers from experiment config.

Follows the same pattern as ``chunking/factory.py`` and
``retrieval/factory.py``: a single ``get_reranker()`` function takes
a typed config object and returns the built component.
"""

from docuseek.config import settings
from docuseek.experiment_config import RerankerConfig
from docuseek.reranking.base import BaseReranker
from docuseek.reranking.colbert import ColBERTReranker
from docuseek.reranking.cross_encoder import CrossEncoderReranker


def get_reranker(config: RerankerConfig) -> BaseReranker | None:
    """Build a reranker from experiment config, or return None if disabled.

    Args:
        config: Reranker section of the experiment config.

    Returns:
        A reranker satisfying the ``BaseReranker`` protocol,
        or ``None`` if ``config.enabled`` is False.

    Raises:
        ValueError: If ``config.method`` is not a recognised reranker.
    """
    if not config.enabled:
        return None

    match config.method:
        case "colbert":
            return ColBERTReranker(model_name=settings.colbert_model_name)
        case "cross_encoder":
            return CrossEncoderReranker(model_name=settings.cross_encoder_model_name)
        case _:
            raise ValueError(
                f"Unknown reranker method: {config.method!r}. Available: colbert, cross_encoder."
            )
