"""
docuseek/embedding/dense.py
----------------------------
Dense embedder backed by sentence-transformers.

Implements asymmetric embedding:
  - embed_documents(): no prefix, used at index-build time
  - embed_queries():   instruction prefix, used at query time
  - embed_query():     inherited default, single-query convenience wrapper

Instruction format expected by Harrier / E5-style models:
    "Instruct: {task_instruction}\\nQuery: {query}"

The default instruction targets passage retrieval. Other instructions
from the MTEB task registry can be injected at construction time
to benchmark different retrieval tasks without changing this class.
"""

import numpy as np
from sentence_transformers import SentenceTransformer

from docuseek.config import settings
from docuseek.embedding.base import BaseEmbedder

_DEFAULT_INSTRUCTION = "Retrieve semantically similar text"


class DenseEmbedder(BaseEmbedder):
    """
    Sentence-transformer embedder with asymmetric query/document encoding.

    Attributes:
        _model:       Loaded SentenceTransformer instance.
        _instruction: Task instruction prepended to all queries.
        _batch_size:  Number of texts per encoding forward pass.
    """

    def __init__(
        self,
        model_name: str = settings.dense_embd_model_name,
        instruction: str = _DEFAULT_INSTRUCTION,
        batch_size: int = 128,
    ) -> None:
        """
        Args:
            model_name: HuggingFace model identifier.
                        Defaults to settings.dense_embd_model_name.
        """
        self._model: SentenceTransformer = SentenceTransformer(model_name)
        self._instruction = instruction
        self._batch_size = batch_size

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """
        Encode document chunks with no instruction prefix.

        Args:
            texts: Raw chunk content strings to encode.

        Returns:
            List of float vectors, one per input text.
        """
        vectors: np.ndarray = self._model.encode(
            texts,
            batch_size=self._batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return vectors.tolist()

    def embed_queries(self, queries: list[str]) -> list[list[float]]:
        """
        Encode queries, each prefixed with the task instruction.

        Args:
            queries: Raw query strings to encode.

        Returns:
            List of float vectors, one per input query.
        """
        prefixed = [f"Instruct: {self._instruction}\nQuery: {q}" for q in queries]
        vectors: np.ndarray = self._model.encode(
            prefixed,
            batch_size=self._batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return vectors.tolist()
