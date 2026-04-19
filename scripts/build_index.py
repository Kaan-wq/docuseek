"""
scripts/build_index.py
----------------------
Wires the full indexing pipeline: load clean docs → chunk → embed → upsert into Qdrant.

This is the script that populates the vector store for the first time.
It is idempotent: re-running with the same collection name and
force_rebuild=False will skip re-indexing if the collection already exists.

Usage
-----
# Build with defaults (fixed chunker, dense embedder)
uv run python scripts/build_index.py

# Force a full rebuild even if the collection exists
uv run python scripts/build_index.py --force

# Choose a specific chunker
uv run python scripts/build_index.py --chunker recursive
uv run python scripts/build_index.py --chunker markdown
"""

import argparse
import dataclasses
import sys
from pathlib import Path

import structlog
from fastembed import SparseTextEmbedding
from fastembed.sparse.sparse_embedding_base import SparseEmbedding
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, Modifier, PointStruct, SparseVectorParams, VectorParams
from rich.console import Console
from rich.progress import track

from docuseek.chunking.base import BaseChunker, Chunk
from docuseek.chunking.document_structure import MarkdownHeaderChunker
from docuseek.chunking.fixed import FixedSizeChunker
from docuseek.chunking.recursive import RecursiveChunker
from docuseek.config import settings
from docuseek.embedding.dense import DenseEmbedder
from docuseek.ingestion.pipeline import load_jsonl
from docuseek.logging import configure_logging

logger = structlog.get_logger()
console = Console()

CHUNKERS: dict[str, type[BaseChunker]] = {
    "fixed": FixedSizeChunker,
    "recursive": RecursiveChunker,
    "markdown": MarkdownHeaderChunker,
}

UPSERT_BATCH_SIZE = 128


def build_index(
    chunker_name: str = "fixed",
    collection_name: str = settings.qdrant_collection_name,
    force_rebuild: bool = False,
) -> None:
    """
    Run the full indexing pipeline.

    Loads clean documents from disk, chunks them, embeds the chunks,
    and upserts the resulting vectors into Qdrant. Creates the collection
    if it does not exist. If force_rebuild=True, drops and recreates it.

    Args:
        chunker_name:    Which chunker to use. One of: fixed, recursive, markdown.
        collection_name: Qdrant collection to upsert into.
        force_rebuild:   Drop and recreate the collection if it already exists.
    """

    if settings.qdrant_cluster_endpoint:
        client = QdrantClient(
            url=settings.qdrant_cluster_endpoint,
            api_key=settings.qdrant_api_key,
        )
    else:
        client = QdrantClient(url=f"http://{settings.qdrant_host}:{settings.qdrant_port}")

    dense_embedder = DenseEmbedder()
    bm25_embedding_model = SparseTextEmbedding("Qdrant/bm25")
    chunker = CHUNKERS[chunker_name]()

    _setup_collection(
        client=client,
        collection_name=collection_name,
        dense_dim=settings.dense_embd_dim,
        late_interaction_dim=settings.late_interaction_embd_dim,
        force_rebuild=force_rebuild,
    )

    # ── 1. Load clean docs from disk ──────────────────────────────────────────
    docs = load_jsonl(Path("data/processed/huggingface.jsonl"))  # TODO: add others when implemented
    logger.info("docs_loaded", source="huggingface", count=len(docs))

    # ── 2. Chunk ──────────────────────────────────────────────────────────────
    all_chunks = [chunk for doc in docs for chunk in chunker.chunk(doc)]
    logger.info("chunks_produced", count=len(all_chunks), chunker=chunker_name)

    # ── 3. Embed + upsert in batches ──────────────────────────────────────────
    for i in track(range(0, len(all_chunks), UPSERT_BATCH_SIZE), description="Embedding"):
        batch = all_chunks[i : i + UPSERT_BATCH_SIZE]

        # ── skip already indexed chunks ───────────
        batch_ids = [chunk.chunk_id for chunk in batch]
        existing = client.retrieve(
            collection_name=collection_name,
            ids=batch_ids,
            with_payload=False,
            with_vectors=False,
        )
        existing_ids = {point.id for point in existing}
        new_chunks = [c for c in batch if c.chunk_id not in existing_ids]
        logger.info("batch_skip", skipped=len(batch) - len(new_chunks), new=len(new_chunks))
        if not new_chunks:
            continue
        # ──────────────────────────────────────────

        dense_embeddings = dense_embedder.embed_documents([c.content for c in batch])
        sparse_embeddings = list(bm25_embedding_model.embed([c.content for c in batch]))
        # TODO late_interaction_embeddings = late_interaction_embedder.embed_documents([c.content for c in batch])
        _upsert_batch(
            client=client,
            collection_name=collection_name,
            chunks=batch,
            dense_vectors=dense_embeddings,
            sparse_vectors=sparse_embeddings,
            # TODO late_interaction_vectors=late_interaction_embeddings,
        )

    logger.info(
        "index_built", collection=collection_name, chunker=chunker_name, chunks=len(all_chunks)
    )


def _setup_collection(
    client: QdrantClient,
    collection_name: str,
    dense_dim: int,
    late_interaction_dim: int,
    force_rebuild: bool = False,
) -> None:
    """
    Create the Qdrant collection if it does not exist.
    Drops and recreates it if force_rebuild=True.

    Args:
        client:               Connected QdrantClient instance.
        collection_name:      Name of the collection to create.
        dense_dim:            Dimension of the dense embeddings.
        late_interaction_dim: Dimension of the late interaction embeddings.
        force_rebuild:        If True, delete the existing collection first.
    """
    exists = client.collection_exists(collection_name)

    if exists and force_rebuild:
        logger.info("dropping_collection", collection=collection_name)
        client.delete_collection(collection_name)
        exists = False

    if not exists:
        client.create_collection(
            collection_name=collection_name,
            vectors_config={
                settings.dense_embd_model_name: VectorParams(
                    size=dense_dim,
                    distance=Distance.COSINE,
                ),
                # TODO : colbert for late interaction and re-ranking
                # "colbertv2.0": VectorParams(
                #   size=late_interaction_dim,
                #   distance=models.Distance.COSINE,
                #   multivector_config=models.MultiVectorConfig(
                #       comparator=models.MultiVectorComparator.MAX_SIM,
                #   ),
                # hnsw_config=models.HnswConfigDiff(m=),
                # ),
            },
            sparse_vectors_config={"bm25": SparseVectorParams(modifier=Modifier.IDF)},
        )
        logger.info("collection_created", collection=collection_name)
    else:
        logger.info("collection_exists_skipping", collection=collection_name)


def _upsert_batch(
    client: QdrantClient,
    collection_name: str,
    chunks: list[Chunk],
    dense_vectors: list[list[float]],
    sparse_vectors: list[SparseEmbedding],
    # TODO late_interaction_vectors: list[list[float]],
) -> None:
    """
    Upsert a single batch of chunks and their vectors into Qdrant.

    Each point's payload carries the full chunk metadata so the retriever
    can reconstruct a Chunk object from a search result without a second lookup.

    Args:
        client:                   Connected QdrantClient instance.
        collection_name:          Target collection name.
        chunks:                   Batch of Chunk objects to upsert.
        dense_vectors:            Dense embedding vectors, one per chunk.
        sparse_vectors:           Sparse embedding vectors, one per chunk.
        late_interaction_vectors: Late interaction embedding vectors, one per chunk.
    """
    # TODO include late_interaction_vectors

    points = []
    for chunk, dense_vector, sparse_vector in zip(
        chunks, dense_vectors, sparse_vectors, strict=True
    ):
        vector: dict = {
            settings.dense_embd_model_name: dense_vector,
            "bm25": sparse_vector.as_object(),
            # TODO: settings.late_interaction_embd_model_name.split("/")[-1]: late_interaction_vector
        }
        points.append(
            PointStruct(
                id=chunk.chunk_id,
                vector=vector,
                payload=dataclasses.asdict(chunk),
            )
        )

    client.upsert(collection_name=collection_name, points=points)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the DocuSeek Qdrant index.")
    parser.add_argument(
        "--chunker",
        choices=list(CHUNKERS.keys()),
        default="fixed",
        help="Chunking strategy to use. Default: fixed.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="Drop and rebuild the collection even if it already exists.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    configure_logging(log_level="info")
    args = parse_args()
    try:
        build_index(chunker_name=args.chunker, force_rebuild=args.force)
    except Exception:
        logger.exception("build_index_failed")
        sys.exit(1)
