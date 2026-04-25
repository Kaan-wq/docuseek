"""
docuseek/query/model.py
------------------------
Shared LLM loading for query rewrite strategies.
"""

import structlog
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from docuseek.config import settings

logger = structlog.get_logger(__name__)


def load_query_model() -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """Load the query LLM and tokenizer.

    Used by ``QueryRewritePipeline`` to share a single model
    instance between HyDE and multi-query rewriters.
    Also used as the standalone fallback inside each rewriter
    when only one strategy is enabled.
    """
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    tokenizer = AutoTokenizer.from_pretrained(settings.query_model_name)
    model = AutoModelForCausalLM.from_pretrained(
        settings.query_model_name,
        dtype=torch.float16,
    ).to(device)
    model.eval()

    logger.info(
        "query_model_loaded",
        model=settings.query_model_name,
        device=device,
    )

    return model, tokenizer
