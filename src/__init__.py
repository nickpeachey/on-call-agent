"""AI On-Call Agent Package

An intelligent automation system for monitoring and resolving ETL infrastructure issues.
"""

__version__ = "0.1.0"
__author__ = "Engineering Team"
__email__ = "engineering@yourcompany.com"

from .core.config import settings
from .core.logging import setup_logging

# Initialize logging when package is imported
setup_logging()

__all__ = ["settings", "setup_logging"]
