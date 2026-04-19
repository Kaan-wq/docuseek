"""
docuseek/api/routes/query.py
-----------------------------
POST /query — end-to-end RAG: retrieve then generate.
"""

from fastapi import APIRouter

from docuseek.api.schemas import QueryRequest, QueryResponse, SourceInfo
from docuseek.embedding.dense import DenseEmbedder
from docuseek.generation.mistral_api import MistralGenerator
from docuseek.retrieval.dense import DenseRetriever

router = APIRouter()

# Instantiated once at import: load a single time, not per request.
_embedder = DenseEmbedder()
_retriever = DenseRetriever(embedder=_embedder)
_generator = MistralGenerator()


@router.post("/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    """
    Run the full RAG pipeline and return a grounded answer.

    Args:
        request: QueryRequest containing the user's question.

    Returns:
        QueryResponse with the generated answer and deduplicated sources.
    """
    chunks = _retriever.retrieve(request.question)
    answer = _generator.generate(request.question, chunks)

    # Dedupe by URL — multiple chunks often come from the same document
    seen: set[str] = set()
    sources: list[SourceInfo] = []
    for chunk in chunks:
        if chunk.doc_url in seen:
            continue
        seen.add(chunk.doc_url)
        sources.append(SourceInfo(title=chunk.doc_title, url=chunk.doc_url, source=chunk.source))

    return QueryResponse(answer=answer, sources=sources)
