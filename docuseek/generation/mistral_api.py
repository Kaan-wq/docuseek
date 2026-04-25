"""
docuseek/generation/mistral_api.py
-----------------------------------
Generator backed by the Mistral API.
"""

from mistralai import Mistral

from docuseek.chunking.base import Chunk
from docuseek.config import settings
from docuseek.experiment_config import GenerationConfig
from docuseek.generation.prompting import PromptAssembler


class MistralGenerator:
    def __init__(
        self,
        model: str = settings.mistral_model,
        assembler: PromptAssembler | None = None,
    ) -> None:
        """
        Args:
            model:     Mistral model identifier. Defaults to settings.mistral_model.
            assembler: Prompt assembler (CoT, Few-Shot, Budget-Forcing).
        """
        self._client = Mistral(api_key=settings.mistral_api_key)
        self._model = model
        self._assembler = assembler or PromptAssembler(GenerationConfig())

    def generate(self, query: str, chunks: list[Chunk]) -> str:
        """
        Generate an answer grounded in the retrieved chunks.

        Formats retrieved chunks into a numbered context block and
        appends the user query.

        Args:
            query:  Raw query string from the user.
            chunks: Retrieved chunks ordered by relevance.

        Returns:
            Generated answer string.
        """
        messages = self._assembler.build(query, chunks)
        response = self._client.chat.complete(model=self._model, messages=messages)
        return response.choices[0].message.content
