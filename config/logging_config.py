import logging
import sys
import structlog
from config.settings import settings

def setup_logging() -> None:
    """Configure structlog for pretty-printed stdout console logs and JSON-formatted agent.json.log."""
    log_level_str = settings.LOG_LEVEL.upper()
    numeric_level = getattr(logging, log_level_str, logging.INFO)

    # Configure root standard logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)
    root_logger.handlers = []

    shared_processors = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.StackInfoRenderer(),
        structlog.dev.set_exc_info,
        structlog.processors.TimeStamper(fmt="%Y-%m-%d %H:%M:%S", utc=False),
    ]

    # Console stream handler
    console_handler = logging.StreamHandler(sys.stdout)
    if settings.ENV.lower() == "development":
        console_formatter = structlog.stdlib.ProcessorFormatter(
            processor=structlog.dev.ConsoleRenderer(colors=True),
            foreign_pre_chain=shared_processors,
        )
    else:
        console_formatter = structlog.stdlib.ProcessorFormatter(
            processor=structlog.processors.JSONRenderer(),
            foreign_pre_chain=shared_processors,
        )
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)

    # File handler for structured log streaming in UI
    file_handler = logging.FileHandler("agent.json.log", mode="w", encoding="utf-8")
    file_formatter = structlog.stdlib.ProcessorFormatter(
        processor=structlog.processors.JSONRenderer(),
        foreign_pre_chain=shared_processors,
    )
    file_handler.setFormatter(file_formatter)
    root_logger.addHandler(file_handler)

    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
