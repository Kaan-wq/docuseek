"""
docuseek/generation/local_generator.py
----------------------------------------
Generator backed by a local HuggingFace model via the transformers pipeline.
"""

from __future__ import annotations

import structlog
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from docuseek.chunking.base import Chunk
from docuseek.config import settings
from docuseek.experiment_config import GenerationConfig
from docuseek.generation.prompting import PromptAssembler

logger = structlog.get_logger(__name__)


class LocalGenerator:
    """Generates answers using a local HuggingFace model.

    Drop-in replacement for MistralGenerator.
    """

    def __init__(
        self,
        model_name: str = settings.local_generation_model_name,
        assembler: PromptAssembler | None = None,
    ) -> None:
        """
        Args:
            model_name: HuggingFace model identifier. Defaults to
                        settings.local_generation_model_name.
            assembler:  Prompt assembler (CoT, few-shot, budget-forcing).
                        Defaults to a baseline GenerationConfig.
        """

        self._assembler = assembler or PromptAssembler(GenerationConfig())
        self._model, self._tokenizer = self._load_model(model_name)

    def generate(self, query: str, chunks: list[Chunk]) -> str:
        """Generate an answer grounded in the retrieved chunks.

        Args:
            query:  Raw query string from the user.
            chunks: Retrieved chunks ordered by relevance.

        Returns:
            Generated answer string.
        """

        messages = self._assembler.build(query, chunks)

        inputs = self._tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            add_generation_prompt=True,
        ).to(self._model.device)

        input_tokens = inputs["input_ids"].shape[-1]

        with torch.inference_mode():
            outputs = self._model.generate(
                inputs["input_ids"],
                max_new_tokens=self._assembler._config.max_tokens,
                temperature=0.3,
                top_p=0.9,
                do_sample=True,
            )

        generated_ids = outputs[0, input_tokens:]
        answer = self._tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

        logger.debug("local_generator_response", length=len(answer))
        return answer

    @staticmethod
    def _load_model(model_name: str) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
        """Load model and tokenizer with sensible defaults for generation."""

        logger.info("local_generator_loading", model_name=model_name)

        tokenizer = AutoTokenizer.from_pretrained(model_name)

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.bfloat16,
            device_map="auto",
        )
        model.eval()

        logger.info("local_generator_loaded", model_name=model_name)
        return model, tokenizer
