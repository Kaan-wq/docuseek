"""
docuseek/generation/prompting.py
----------------------------------
Prompt assembly strategies for generation.

Strategies are composable flags in GenerationConfig and applied in order:
    1. Few-shot       — prepend example Q/A pairs to guide response style.
    2. CoT            — append a reasoning instruction before the answer.
    3. Budget-forcing — constrain response length via explicit token budget.
"""

from __future__ import annotations

from docuseek.chunking.base import Chunk
from docuseek.experiment_config import GenerationConfig

Message = dict[str, str]

_BASE_SYSTEM_PROMPT = (
    "You are a helpful assistant specialising in ML framework documentation "
    "(HuggingFace, PyTorch, PEFT, etc.). Answer the user's question using "
    "only the provided documentation excerpts. If the answer is not in the "
    "excerpts, say so."
)


_COT_SUFFIX = "Think step by step before giving your final answer."
_BUDGET_TEMPLATE = "Answer in {max_tokens} tokens or fewer."
_FEW_SHOT_EXAMPLES: list[dict[str, str]] = [
    # Each entry: {"question": ..., "context": ..., "answer": ...}
    # TODO: populate with 2-3 representative Q/A pairs from the gold set
]


class PromptAssembler:
    """Assembles a messages list from retrieved chunks and a query.

    Strategies are composed in this order:
        few-shot  → base prompt + examples
        CoT       → reasoning instruction appended to user turn
        budget    → token budget constraint appended to user turn
    """

    def __init__(self, config: GenerationConfig) -> None:
        self._config = config

    def build(self, query: str, chunks: list[Chunk]) -> list[Message]:
        """Assemble the full messages list for the generator.

        Args:
            query:  Raw user question.
            chunks: Retrieved chunks, already ordered by relevance.

        Returns:
            OpenAI-style messages list ready to pass to the generator.
        """
        system = self._build_system()
        user = self._build_user(query, chunks)
        messages: list[Message] = []

        if self._config.few_shot:
            messages.extend(self._few_shot_messages())

        messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user})

        return messages

    def _build_system(self) -> str:
        """Return the system prompt, unmodified for now."""
        return _BASE_SYSTEM_PROMPT

    def _build_user(self, query: str, chunks: list[Chunk]) -> str:
        """Assemble the user turn: context block + query + optional suffixes."""
        context = self._format_context(chunks)
        user = f"{context}\n\nQuestion: {query}"

        if self._config.cot:
            user = f"{user}\n\n{_COT_SUFFIX}"

        if self._config.budget_forcing:
            budget = _BUDGET_TEMPLATE.format(max_tokens=self._config.max_tokens)
            user = f"{user}\n\n{budget}"

        return user

    def _format_context(self, chunks: list[Chunk]) -> str:
        """Format retrieved chunks as a numbered context block."""
        # Placeholder — numbered excerpts, one per chunk
        lines = [f"[{i}] {chunk.content}" for i, chunk in enumerate(chunks, start=1)]
        return "\n\n".join(lines)

    def _few_shot_messages(self) -> list[Message]:
        """Return few-shot Q/A pairs as alternating user/assistant messages."""
        messages: list[Message] = []
        for example in _FEW_SHOT_EXAMPLES:
            messages.append({"role": "user", "content": example["question"]})
            messages.append({"role": "assistant", "content": example["answer"]})
        return messages
