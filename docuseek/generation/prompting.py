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

_BASE_SYSTEM_PROMPT = """\
You are a precise technical assistant specialising in ML framework documentation.
Answer the user's question using only the provided context passages.
If the answer is not present in the context, say so explicitly — do not speculate.
When referencing specific information, mention the source title it came from.\
"""


_COT_SUFFIX = "Think step by step before giving your final answer."
_BUDGET_TEMPLATE = "Answer in {max_tokens} tokens or fewer."
_FEW_SHOT_EXAMPLES: list[dict[str, str]] = [
    {
        "question": "What platforms and devices does the Transformers kernel system support, and what is the minimum NVIDIA compute capability required?",
        "answer": "The kernel system supports four platforms: NVIDIA GPUs via CUDA (requiring compute capability 7.0 or higher, covering Volta, Turing, Ampere, Hopper, and Blackwell architectures), AMD GPUs via ROCm-supported devices, Apple Silicon M-series chips (M1 through M4 and newer) via Metal, and Intel Data Center GPU Max Series and compatible devices via XPU. Precompiled binaries for these platforms are distributed through the Hub's kernels-community organization and detected automatically at runtime.",
        "context": "| platform | supported devices |\n| :--- | :--- |\n| NVIDIA GPUs (CUDA) | Modern architectures with compute capability 7.0+ (Volta, Turing, Ampere, Hopper, Blackwell) |\n| AMD GPUs (ROCm) | Compatible with ROCm-supported devices |\n| Apple Silicon (Metal) | M-series chips (M1, M2, M3, M4 and newer) |\n| Intel GPUs (XPU) | Intel Data Center GPU Max Series and compatible devices |\n\n[Kernels](https://huggingface.co/docs/kernels/index) solves this by distributing precompiled binaries through the [Hub](https://huggingface.co/kernels-community). It detects your platform at runtime and loads the right binary automatically.",
    },
    {
        "question": "How do you enable Liger Kernel optimizations when training with Transformers?",
        "answer": "Install the library with `pip install liger-kernel`, then set `use_liger_kernel=True` in `TrainingArguments`. This patches the corresponding model layers with Liger's fused Triton kernels.",
        "context": "## Liger\n\n[Liger Kernel](https://github.com/linkedin/Liger-Kernel) fuses layers like RMSNorm, RoPE, SwiGLU, CrossEntropy, and FusedLinearCrossEntropy into single Triton kernels. It's compatible with FlashAttention, FSDP, and DeepSpeed.\n\n```bash\npip install liger-kernel\n```\n\nSet `use_liger_kernel=True` in [`TrainingArguments`] to patch the corresponding model layers with Liger's kernels.\n\n```py\nfrom transformers import TrainingArguments\n\ntraining_args = TrainingArguments(\n    ...,\n    use_liger_kernel=True\n)\n```",
    },
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
        user = self._build_user(query, chunks)
        messages: list[Message] = []
        messages.append({"role": "system", "content": _BASE_SYSTEM_PROMPT})
        if self._config.few_shot:
            messages.extend(self._few_shot_messages())
        messages.append({"role": "user", "content": user})

        return messages

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
        lines = [
            f"[{i}] {chunk.doc_title}\n{chunk.content}\nSource: {chunk.doc_url}\nLibrary: {chunk.source}"
            for i, chunk in enumerate(chunks, start=1)
        ]
        return "## Context\n\n" + "\n\n---\n\n".join(lines)

    def _few_shot_messages(self) -> list[Message]:
        """Return few-shot Q/A pairs as alternating user/assistant messages."""
        messages: list[Message] = []
        for example in _FEW_SHOT_EXAMPLES:
            messages.append({"role": "user", "content": example["question"]})
            messages.append({"role": "assistant", "content": example["answer"]})
        return messages
