# docuseek/logging.py

import logging

import structlog


def configure_logging(log_level: str = "info") -> None:
    level = getattr(logging, log_level.upper())

    logging.basicConfig(level=level, format="%(message)s")

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="%H:%M:%S"),
            structlog.dev.ConsoleRenderer(
                colors=True,
                exception_formatter=structlog.dev.better_traceback,
                pad_event=32,
            ),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
    )
