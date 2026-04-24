"""
docuseek/query/multi_query.py
------------------------------
Multi-query expansion: generate multiple phrasings of the same question.

Different phrasings activate different regions of embedding space.
A question about "fine-tuning" might miss documentation that says
"transfer learning" or "adapter training".  Multi-query generates
several variants, retrieves for each independently, and merges the
results — improving recall at the cost of N retrieval calls.

The model and tokenizer can be injected by ``QueryRewritePipeline``
to share a single instance with other LLM-based rewriters (e.g.
HyDE), avoiding duplicate parameter loads.

Example::

    "How do I fine-tune a model with LoRA?"
    → [
        "How do I fine-tune a model with LoRA?",
        "What is the process for LoRA adapter training in PEFT?",
        "How to apply low-rank adaptation for transfer learning?",
      ]
"""

import time

import structlog
import torch
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from docuseek.eval.query_metrics import QueryMethodSample
from docuseek.query.model import load_query_model

logger = structlog.get_logger(__name__)

_NUM_VARIANTS = 3

_MULTI_QUERY_SYSTEM_PROMPT = (
    "You are a search query generator for ML framework documentation "
    "(HuggingFace, PyTorch, PEFT, etc.). Given a user question, "
    f"generate {_NUM_VARIANTS} alternative phrasings that would help find relevant "
    "documentation. Each variant should use different terminology "
    "while preserving the original intent. "
    f"Output exactly {_NUM_VARIANTS} queries, one per line, with no numbering, "
    "no bullet points, and no extra text."
)


class MultiQueryRewriter:
    """Expand a query into multiple phrasings for broader retrieval."""

    def __init__(
        self,
        model: PreTrainedModel | None = None,
        tokenizer: PreTrainedTokenizerBase | None = None,
    ) -> None:
        if (model is None) != (tokenizer is None):
            msg = "model and tokenizer must be provided together or both omitted"
            raise ValueError(msg)

        if model is not None:
            self._model = model
            self._tokenizer = tokenizer
        else:
            self._model, self._tokenizer = load_query_model()

    @torch.inference_mode()
    def rewrite(self, query: str) -> list[str]:
        """Generate query variants and return them with the original.

        Args:
            query: Raw user question.

        Returns:
            List starting with the original query followed by up to
            ``_NUM_VARIANTS`` alternative phrasings.  Falls back to
            ``[query]`` if the LLM output is malformed.
        """
        messages = [
            {"role": "system", "content": _MULTI_QUERY_SYSTEM_PROMPT},
            {"role": "user", "content": query},
        ]

        inputs = self._tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            add_generation_prompt=True,
        ).to(self._model.device)

        outputs = self._model.generate(
            inputs,
            max_new_tokens=150,
            temperature=0.8,
            top_p=0.9,
            do_sample=True,
        )

        generated_ids = outputs[0, inputs.shape[-1] :]
        raw_output = self._tokenizer.decode(
            generated_ids,
            skip_special_tokens=True,
        ).strip()

        # Parse one query per line, discard empty or whitespace-only lines
        variants = [line.strip() for line in raw_output.splitlines() if line.strip()][
            :_NUM_VARIANTS
        ]

        if not variants:
            logger.warning("multi_query_empty_output", original=query, raw=raw_output)
            return [query]

        logger.debug(
            "multi_query_rewrite",
            original=query,
            variants=len(variants),
        )

        return [query, *variants]

    @torch.inference_mode()
    def rewrite_timed(self, query: str) -> tuple[list[str], QueryMethodSample]:
        """Generate query variants and return cost metrics."""
        messages = [
            {"role": "system", "content": _MULTI_QUERY_SYSTEM_PROMPT},
            {"role": "user", "content": query},
        ]
        inputs = self._tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            add_generation_prompt=True,
        ).to(self._model.device)

        input_tokens = inputs.shape[-1]

        t0 = time.perf_counter()
        outputs = self._model.generate(
            inputs,
            max_new_tokens=150,
            temperature=0.8,
            top_p=0.9,
            do_sample=True,
        )
        latency_ms = (time.perf_counter() - t0) * 1000

        generated_ids = outputs[0, inputs.shape[-1] :]
        raw_output = self._tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

        variants = [line.strip() for line in raw_output.splitlines() if line.strip()][
            :_NUM_VARIANTS
        ]

        if not variants:
            logger.warning("multi_query_empty_output", original=query, raw=raw_output)
            variants = []

        logger.debug("multi_query_rewrite", original=query, variants=len(variants))

        result = [query, *variants]
        return result, QueryMethodSample(
            method="multi_query",
            latency_ms=latency_ms,
            input_tokens=input_tokens,
            output_tokens=len(generated_ids),
            original=query,
            variants=tuple(variants),
        )
