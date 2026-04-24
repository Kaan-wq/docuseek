"""
docuseek/reranking/colbert.py
------------------------------
ColBERT reranker using late-interaction (MaxSim) scoring.
"""

import time

import structlog
import torch
from torch.nn.functional import normalize
from transformers import AutoModel, AutoTokenizer

from docuseek.chunking.base import Chunk

logger = structlog.get_logger(__name__)

_QUERY_MARKER = "[QueryMarker]"
_DOC_MARKER = "[DocumentMarker]"
_MAX_LENGTH = 512


class ColBERTReranker:
    """Reranks chunks using ColBERT-style MaxSim scoring."""

    def __init__(self, model_name: str, device: str | None = None) -> None:
        self._device = device or (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )

        self._tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
        )
        self._model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
        )
        self._model.to(self._device)
        self._model.eval()

        logger.info("colbert_reranker_loaded", model=model_name, device=self._device)

    @torch.inference_mode()
    def _encode_tokens(
        self,
        texts: list[str],
        marker: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode texts into per-token embeddings with a role marker.

        Returns:
            ``(embeddings, attention_mask)`` where embeddings is
            ``(batch, seq_len, dim)`` with padding positions zeroed out.
        """
        prefixed = [f"{marker} {text}" for text in texts]

        inputs = self._tokenizer(
            prefixed,
            padding=True,
            truncation=True,
            max_length=_MAX_LENGTH,
            return_tensors="pt",
        ).to(self._device)

        outputs = self._model(**inputs)
        token_embeddings = outputs.last_hidden_state  # (batch, seq_len, dim)

        mask = inputs["attention_mask"].unsqueeze(-1)  # (batch, seq_len, 1)
        token_embeddings = token_embeddings * mask

        return token_embeddings, inputs["attention_mask"]

    def _maxsim_score(
        self,
        query_vectors: torch.Tensor,
        doc_vectors: torch.Tensor,
    ) -> float:
        """Compute MaxSim between one query and one document.

        For each query token, find the max cosine similarity across all document
        tokens, then sum. ``query_vectors`` is ``(query_len, dim)``,
        ``doc_vectors`` is ``(doc_len, dim)``.
        """
        q = normalize(query_vectors, dim=-1)
        d = normalize(doc_vectors, dim=-1)
        sim = q @ d.T  # (query_len, doc_len)
        return sim.max(dim=-1).values.sum().item()

    @torch.inference_mode()
    def rerank(self, query: str, chunks: list[Chunk], top_k: int = 10) -> list[Chunk]:
        """Reorder candidate chunks using ColBERT MaxSim scoring.

        Args:
            query:  Raw query string.
            chunks: Candidate chunks from first-stage retrieval.
            top_k:  Number of chunks to return.

        Returns:
            Top-k chunks ordered by descending MaxSim score.
        """
        if not chunks or len(chunks) <= top_k:
            return chunks

        query_emb, _ = self._encode_tokens([query], _QUERY_MARKER)
        query_emb = query_emb.squeeze(0)  # (query_len, dim)

        doc_embs, doc_masks = self._encode_tokens(
            [c.content for c in chunks],
            _DOC_MARKER,
        )

        scores: list[float] = []
        for i in range(len(chunks)):
            mask_len = doc_masks[i].sum().item()
            doc_emb = doc_embs[i, :mask_len]  # (actual_len, dim)
            scores.append(self._maxsim_score(query_emb, doc_emb))

        scored_chunks = sorted(
            zip(chunks, scores, strict=True),
            key=lambda x: x[1],
            reverse=True,
        )

        logger.debug(
            "colbert_reranked",
            candidates=len(chunks),
            top_k=top_k,
            top_score=scored_chunks[0][1],
        )

        return [chunk for chunk, _ in scored_chunks[:top_k]]

    @torch.inference_mode()
    def rerank_timed(
        self, query: str, chunks: list[Chunk], top_k: int = 10
    ) -> tuple[list[Chunk], float]:
        """Rerank chunks and return wall-clock latency in milliseconds."""
        t0 = time.perf_counter()
        result = self.rerank(query, chunks, top_k)
        if self._device == "cuda":
            torch.cuda.synchronize()
        return result, (time.perf_counter() - t0) * 1000
