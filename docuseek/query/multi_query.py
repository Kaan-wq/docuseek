"""
docuseek/query/multi_query.py
------------------------------
Multi-query expansion: generate multiple phrasings of the same question.

Different phrasings activate different regions of embedding space.
A question about "fine-tuning" might miss documentation that says
"transfer learning" or "adapter training".

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
from transformers import PreTrainedModel, PreTrainedTokenizerBase

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
    """Expands a query into multiple phrasings for broader retrieval."""

    def __init__(
        self,
        model: PreTrainedModel | None = None,
        tokenizer: PreTrainedTokenizerBase | None = None,
    ) -> None:
        if (model is None) != (tokenizer is None):
            raise ValueError("model and tokenizer must be provided together or both omitted")

        self._model, self._tokenizer = (
            (model, tokenizer) if model is not None else load_query_model()
        )

    @torch.inference_mode()
    def rewrite(self, query: str) -> list[str]:
        """Generate query variants and return them with the original.

        Args:
            query: Raw user question.

        Returns:
            List starting with the original query followed by up to
            ``_NUM_VARIANTS`` alternative phrasings. Falls back to
            ``[query]`` if the LLM output is malformed.
        """
        result, _ = self._rewrite_inner(query)
        return result

    @torch.inference_mode()
    def rewrite_timed(self, query: str) -> tuple[list[str], QueryMethodSample]:
        """Generate query variants and return cost metrics."""
        return self._rewrite_inner(query)

    @torch.inference_mode()
    def _rewrite_inner(self, query: str) -> tuple[list[str], QueryMethodSample]:
        messages = [
            {"role": "system", "content": _MULTI_QUERY_SYSTEM_PROMPT},
            {"role": "user", "content": query},
        ]

        inputs = self._tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            add_generation_prompt=True,
        ).to(self._model.device)

        input_ids = inputs["input_ids"]
        input_tokens = input_ids.shape[-1]

        t0 = time.perf_counter()
        outputs = self._model.generate(
            input_ids,
            max_new_tokens=150,
            temperature=0.8,
            top_p=0.9,
            do_sample=True,
        )
        latency_ms = (time.perf_counter() - t0) * 1000

        generated_ids = outputs[0, input_tokens:]
        raw_output = self._tokenizer.decode(
            generated_ids,
            skip_special_tokens=True,
        ).strip()

        variants = [line.strip() for line in raw_output.splitlines() if line.strip()][
            :_NUM_VARIANTS
        ]

        if not variants:
            logger.warning("multi_query_empty_output", original=query, raw=raw_output)

        logger.debug("multi_query_rewrite", original=query, variants=len(variants))

        return [query, *variants], QueryMethodSample(
            method="multi_query",
            latency_ms=latency_ms,
            input_tokens=input_tokens,
            output_tokens=len(generated_ids),
            original=query,
            variants=tuple(variants),
        )
