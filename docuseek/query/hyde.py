"""
docuseek/query/hyde.py
-----------------------
HyDE (Hypothetical Document Embeddings) query rewriter.

The hypothetical only needs to use the right vocabulary and
phrasing to land near the real documents in embedding space.
"""

import time

import structlog
import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from docuseek.eval.query_metrics import QueryMethodSample
from docuseek.query.model import load_query_model

logger = structlog.get_logger(__name__)

_HYDE_SYSTEM_PROMPT = (
    "You are a technical documentation writer for ML frameworks "
    "(HuggingFace, PyTorch, PEFT, etc.). Given a user question, "
    "write a short documentation paragraph that would answer it. "
    "Be specific with class names, method signatures, and parameters. "
    "Do not hedge or say 'I don't know'. Write as if this paragraph "
    "exists in the official documentation."
)


class HyDEQueryRewriter:
    """Generates a hypothetical answer document and uses it as the retrieval query."""

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
        """Generate a hypothetical document and return it alongside the original.

        Args:
            query: Raw user question.

        Returns:
            ``[original_query, hypothetical_doc]``. The retriever runs both
            and merges results via RRF.
        """
        variants, _ = self._rewrite_inner(query)
        return variants

    @torch.inference_mode()
    def rewrite_timed(self, query: str) -> tuple[list[str], QueryMethodSample]:
        """Generate a hypothetical document and return cost metrics."""
        return self._rewrite_inner(query)

    @torch.inference_mode()
    def _rewrite_inner(self, query: str) -> tuple[list[str], QueryMethodSample]:
        messages = [
            {"role": "system", "content": _HYDE_SYSTEM_PROMPT},
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
            max_new_tokens=200,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
        )
        latency_ms = (time.perf_counter() - t0) * 1000

        generated_ids = outputs[0, input_tokens:]
        hypothetical_doc = self._tokenizer.decode(
            generated_ids,
            skip_special_tokens=True,
        ).strip()

        logger.debug("hyde_rewrite", original=query, hypothetical_length=len(hypothetical_doc))

        return [query, hypothetical_doc], QueryMethodSample(
            method="hyde",
            latency_ms=latency_ms,
            input_tokens=input_tokens,
            output_tokens=len(generated_ids),
            original=query,
            variants=(hypothetical_doc,),
        )
