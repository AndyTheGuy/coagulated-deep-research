import structlog
from config.settings import settings
from config.logging_config import setup_logging

# Initialize logging system immediately on import
setup_logging()

# Export main settings and default logger instance
logger = structlog.get_logger("deep-research")

__all__ = ["settings", "logger"]
