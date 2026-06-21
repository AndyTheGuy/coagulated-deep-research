import logging
import sys
import structlog
from config.settings import settings

def setup_logging() -> None:
    """Configure structlog for either pretty-printed or structured JSON logs."""
    log_level_str = settings.LOG_LEVEL.upper()
    numeric_level = getattr(logging, log_level_str, logging.INFO)

    # Configure base logging to output directly to stdout
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=numeric_level,
    )

    shared_processors = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.StackInfoRenderer(),
        structlog.dev.set_exc_info,
        structlog.processors.TimeStamper(fmt="%Y-%m-%d %H:%M:%S", utc=False),
    ]

    # Console renderer in dev mode, JSON renderer in prod mode
    if settings.ENV.lower() == "development":
        processors = shared_processors + [
            structlog.dev.ConsoleRenderer(colors=True)
        ]
    else:
        processors = shared_processors + [
            structlog.processors.dict_tracebacks,
            structlog.processors.JSONRenderer()
        ]

    structlog.configure(
        processors=processors,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
