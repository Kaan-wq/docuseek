# docuseek/logging.py
import logging

import structlog


def configure_logging(log_level: str = "info") -> None:
    level = getattr(logging, log_level.upper())

    # Silence all third-party loggers at root level, then bring docuseek back up
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    logging.getLogger("docuseek").setLevel(level)

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
