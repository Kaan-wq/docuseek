"""
docuseek/api/main.py
---------------------
FastAPI application factory.

Components are loaded once at startup via the lifespan context manager and
stored on app.state. Routes read from app.state — no module-level globals.

The experiment config path is read from the DOCUSEEK_CONFIG_PATH environment
variable, set by scripts/serve.py.

    app.state.config     — ExperimentConfig
    app.state.retriever  — BaseRetriever
    app.state.reranker   — BaseReranker | None
    app.state.query      — QueryRewritePipeline
    app.state.generator  — MistralGenerator
    app.state.tracer     — LangfuseTracer
"""

from __future__ import annotations

import os
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from docuseek.api.routes import health, query
from docuseek.experiment_config import ExperimentConfig
from docuseek.generation.mistral_api import MistralGenerator
from docuseek.generation.prompting import PromptAssembler
from docuseek.logging import configure_logging
from docuseek.observability.langfuse_tracer import LangfuseTracer
from docuseek.query.rewrite import QueryRewritePipeline
from docuseek.reranking.factory import get_reranker
from docuseek.retrieval.factory import get_retriever

configure_logging(log_level="info")
logger = structlog.get_logger()

_CONFIG_ENV_VAR = "DOCUSEEK_CONFIG_PATH"


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Load all pipeline components once at startup, clean up on shutdown."""
    config_path_str = os.environ.get(_CONFIG_ENV_VAR)
    if not config_path_str:
        raise RuntimeError(
            f"Environment variable {_CONFIG_ENV_VAR} is not set. "
            "Start the server via scripts/serve.py --config PATH."
        )

    config_path = Path(config_path_str)
    if not config_path.exists():
        raise RuntimeError(f"Config not found: {config_path}")

    config = ExperimentConfig.from_yaml(config_path)
    logger.info("config_loaded", experiment=config.name)

    app.state.config = config
    app.state.retriever = get_retriever(config.retriever)
    app.state.reranker = get_reranker(config.reranker)
    app.state.query = QueryRewritePipeline(config.query)
    app.state.generator = MistralGenerator(assembler=PromptAssembler(config.generation))
    app.state.tracer = LangfuseTracer(experiment_name=config.name)

    logger.info(
        "components_loaded",
        retriever=config.retriever.mode,
        reranker=config.reranker.method if config.reranker.enabled else None,
        experiment=config.name,
    )

    yield

    # Shutdown — flush any pending Langfuse traces
    app.state.tracer.flush()
    logger.info("shutdown_complete")


def create_app() -> FastAPI:
    app = FastAPI(
        title="DocuSeek",
        description="RAG over ML framework documentation.",
        version="0.1.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # tighten if deploying beyond localhost
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    app.include_router(health.router, prefix="/health", tags=["health"])
    app.include_router(query.router, tags=["query"])

    return app


app = create_app()
