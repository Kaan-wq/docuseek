# docuseek/logging.py
import logging

import structlog

_NOISY_LOGGERS = {
    "langchain": logging.WARNING,
    "langchain_text_splitters": logging.WARNING,
    "httpx": logging.WARNING,
    "httpcore": logging.WARNING,
    "urllib3": logging.WARNING,
    "sentence_transformers": logging.WARNING,
    "fastembed": logging.WARNING,
    "qdrant_client": logging.WARNING,
    "transformers": logging.WARNING,
}


def configure_logging(log_level: str = "info") -> None:
    level = getattr(logging, log_level.upper())
    logging.basicConfig(level=level, format="%(message)s")

    # Silence third-party noise — only their warnings and errors come through
    for name, silent_level in _NOISY_LOGGERS.items():
        logging.getLogger(name).setLevel(silent_level)

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.dev.ConsoleRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
    )
