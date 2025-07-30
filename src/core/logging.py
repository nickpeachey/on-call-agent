"""Logging configuration."""

import logging
import sys
from typing import Any, Dict, Optional
import structlog
from rich.logging import RichHandler
from rich.console import Console

from .config import settings


def setup_logging() -> None:
    """Configure application logging."""
    
    # Configure standard library logging
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper()),
        format="%(message)s",
        datefmt="[%X]",
        handlers=[
            RichHandler(
                console=Console(stderr=True),
                show_time=True,
                show_path=True,
                markup=True,
                rich_tracebacks=True,
            )
        ],
    )
    
    # Configure structlog
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer() if not settings.debug else structlog.dev.ConsoleRenderer(),
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Get a structured logger instance."""
    return structlog.get_logger(name)


class LoggerMixin:
    """Mixin to add logging capabilities to classes."""
    
    @property
    def logger(self) -> structlog.stdlib.BoundLogger:
        """Get logger instance for this class."""
        return get_logger(self.__class__.__name__)


def log_function_call(func_name: str, **kwargs: Any) -> None:
    """Log function call with parameters."""
    logger = get_logger("function_call")
    logger.info(f"Calling {func_name}", **kwargs)


def log_error(error: Exception, context: Optional[Dict[str, Any]] = None) -> None:
    """Log error with context."""
    logger = get_logger("error")
    logger.error(
        "Error occurred",
        error=str(error),
        error_type=error.__class__.__name__,
        context=context or {},
        exc_info=True,
    )


def log_info_unless_quiet(logger: structlog.stdlib.BoundLogger, message: str, **kwargs) -> None:
    """Log info message only if not in quiet mode."""
    if not settings.quiet_mode:
        logger.info(message, **kwargs)


def log_activity(logger: structlog.stdlib.BoundLogger, message: str, **kwargs) -> None:
    """Log only important activity - respects quiet mode for routine operations."""
    # Always log errors, warnings, and incidents
    # In quiet mode, skip routine operational messages
    if not settings.quiet_mode or any(keyword in message.lower() for keyword in ['error', 'warning', 'incident', 'failed', 'critical']):
        logger.info(message, **kwargs)
