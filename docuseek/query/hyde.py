"""
docuseek/query/hyde.py
-----------------------
HyDE (Hypothetical Document Embeddings) query rewriter.

Instead of embedding the raw question, HyDE asks an LLM to generate
a hypothetical document that *would* answer the question, then uses
that document as the retrieval query.  This bridges the vocabulary
gap between questions and documentation.

The hypothetical answer does not need to be correct — it only needs
to use the right vocabulary and phrasing to land near the real
documents in embedding space.

The model and tokenizer can be injected by ``QueryRewritePipeline``
to share a single instance with other LLM-based rewriters (e.g.
multi-query), avoiding duplicate 3.8B parameter loads.

Reference: Gao et al., "Precise Zero-Shot Dense Retrieval without
Relevance Labels" (2022).  https://arxiv.org/abs/2212.10496
"""

import structlog
import torch
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

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
    """Generate a hypothetical answer and use it as the retrieval query."""

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
        """Generate a hypothetical document and return it with the original.

        Args:
            query: Raw user question.

        Returns:
            Two-element list: ``[original_query, hypothetical_doc]``.
            The retriever runs both and merges results.
        """
        messages = [
            {"role": "system", "content": _HYDE_SYSTEM_PROMPT},
            {"role": "user", "content": query},
        ]

        inputs = self._tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            add_generation_prompt=True,
        ).to(self._model.device)

        outputs = self._model.generate(
            inputs,
            max_new_tokens=200,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
        )

        # Decode only the generated tokens, not the prompt
        generated_ids = outputs[0, inputs.shape[-1] :]
        hypothetical_doc = self._tokenizer.decode(
            generated_ids,
            skip_special_tokens=True,
        ).strip()

        logger.debug(
            "hyde_rewrite",
            original=query,
            hypothetical_length=len(hypothetical_doc),
        )

        return [query, hypothetical_doc]
