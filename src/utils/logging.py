"""
Logging configuration for the e-commerce RAG pipeline.
"""
import sys
from typing import Optional
from loguru import logger
from pathlib import Path

from .config import Config


def setup_logging(config: Optional[Config] = None) -> None:
    """
    Setup logging configuration using loguru.

    Args:
        config: Optional configuration object
    """
    if config is None:
        config = Config()

    # Remove default logger
    logger.remove()

    # Console logging
    logger.add(
        sys.stdout,
        level=config.log_level.upper(),
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
               "<level>{level: <8}</level> | "
               "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
               "<level>{message}</level>",
        colorize=True
    )

    # File logging
    if config.log_file_path:
        logger.add(
            config.log_file_path,
            level=config.log_level.upper(),
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}",
            rotation=config.log_rotation,
            retention=config.log_retention,
            compression="zip"
        )

    # Set specific loggers to WARNING to reduce noise
    noisy_loggers = [
        "urllib3.connectionpool",
        "transformers.tokenization_utils_base",
        "transformers.configuration_utils",
        "transformers.modeling_utils",
        "PIL.PngImagePlugin",
        "httpx",
        "httpcore"
    ]

    for logger_name in noisy_loggers:
        logger.disable(logger_name)

    logger.info(f"Logging configured - Level: {config.log_level.upper()}")


def get_logger(name: str):
    """Get a logger instance for a specific module."""
    return logger.bind(name=name)


class LoggerMixin:
    """Mixin class to add logging capabilities to any class."""

    @property
    def logger(self):
        """Get logger for this class."""
        return logger.bind(name=self.__class__.__name__)