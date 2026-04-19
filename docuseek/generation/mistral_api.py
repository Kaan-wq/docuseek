"""
docuseek/generation/mistral_api.py
-----------------------------------
Generator backed by the Mistral API.
"""

from mistralai import Mistral

from docuseek.chunking.base import Chunk
from docuseek.config import settings
from docuseek.generation.base import BaseGenerator

_SYSTEM_PROMPT = """\
You are a precise technical assistant specialising in ML framework documentation.
Answer the user's question using only the provided context passages.
If the answer is not present in the context, say so explicitly — do not speculate.
When referencing specific information, mention the source title it came from.\
"""


def _format_context(chunks: list[Chunk]) -> str:
    """
    Format retrieved chunks into a numbered context block for the prompt.

    Each passage includes its index, source title, content, and URL so
    the model can ground its answer and attribute claims to sources.
    The numbered format is intentional — Week 4 will use these indices
    for inline citation generation without changing this structure.

    Args:
        chunks: Retrieved chunks ordered by relevance.

    Returns:
        Formatted multi-line string ready for inclusion in the prompt.
    """
    passages = []
    for i, chunk in enumerate(chunks, start=1):
        passages.append(
            f"[{i}] {chunk.doc_title}\n{chunk.content}\nSource: {chunk.doc_url}\nLibrary: {chunk.source}"
        )
    return "## Context\n\n" + "\n\n---\n\n".join(passages)


class MistralGenerator:
    def __init__(self, model: str = settings.mistral_model) -> None:
        """
        Args:
            model: Mistral model identifier. Defaults to settings.mistral_model.
        """
        self._client = Mistral(api_key=settings.mistral_api_key)
        self._model = model

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
        context = _format_context(chunks)
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": f"{context}\n\n## Query\n\n{query}"},
        ]
        response = self._client.chat.complete(model=self._model, messages=messages)
        return response.choices[0].message.content


_: BaseGenerator = MistralGenerator()
