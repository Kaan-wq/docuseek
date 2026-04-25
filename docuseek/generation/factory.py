"""
docuseek/generation/factory.py
--------------------------------
Factory for generator components.

Returns None when generate_answers is False so benchmark.py never
pays model loading or API costs on retrieval-only experiments.
"""

from __future__ import annotations

import structlog

from docuseek.experiment_config import GenerationConfig
from docuseek.generation.local_generator import LocalGenerator
from docuseek.generation.mistral_api import MistralGenerator
from docuseek.generation.prompting import PromptAssembler

logger = structlog.get_logger(__name__)


def get_generator(
    config: GenerationConfig,
) -> object | None:
    """Build and return the configured generator, or None if disabled.

    Args:
        config: Generation section of the experiment config.

    Returns:
        MistralGenerator, LocalGenerator, or None.
    """
    if not config.generate_answers:
        logger.info("generator_disabled")
        return None

    assembler = PromptAssembler(config)

    if config.backend == "mistral":
        logger.info("generator_backend", backend="mistral")
        return MistralGenerator(assembler=assembler)

    if config.backend == "local":
        logger.info("generator_backend", backend="local")
        return LocalGenerator(assembler=assembler)

    raise ValueError(f"Unknown generation backend: {config.backend!r}")
