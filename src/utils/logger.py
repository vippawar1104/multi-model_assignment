"""
Logger utility module using Loguru for structured logging.
Provides consistent logging across the application.
"""

import sys
from pathlib import Path
from typing import Optional
from loguru import logger


class Logger:
    """
    Centralized logger configuration for the Multi-Modal RAG system.
    """
    
    _initialized = False
    
    @classmethod
    def setup(
        cls,
        log_file: Optional[str] = "logs/app.log",
        level: str = "INFO",
        rotation: str = "100 MB",
        retention: str = "30 days",
        format_string: Optional[str] = None
    ):
        """
        Setup logger with file and console handlers.
        
        Args:
            log_file: Path to log file
            level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            rotation: When to rotate log file
            retention: How long to keep old logs
            format_string: Custom format string
        """
        if cls._initialized:
            return logger
        
        # Remove default handler
        logger.remove()
        
        # Default format
        if format_string is None:
            format_string = (
                "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
                "<level>{level: <8}</level> | "
                "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
                "<level>{message}</level>"
            )
        
        # Console handler with colors
        logger.add(
            sys.stdout,
            format=format_string,
            level=level,
            colorize=True,
            backtrace=True,
            diagnose=True
        )
        
        # File handler
        if log_file:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            logger.add(
                log_file,
                format=format_string,
                level=level,
                rotation=rotation,
                retention=retention,
                compression="zip",
                backtrace=True,
                diagnose=True,
                enqueue=True  # Thread-safe
            )
        
        cls._initialized = True
        logger.info(f"Logger initialized with level: {level}")
        
        return logger
    
    @classmethod
    def get_logger(cls, name: Optional[str] = None):
        """
        Get logger instance.
        
        Args:
            name: Logger name (usually __name__)
            
        Returns:
            Logger instance
        """
        if not cls._initialized:
            cls.setup()
        
        if name:
            return logger.bind(name=name)
        return logger


def get_logger(name: Optional[str] = None):
    """
    Convenience function to get logger.
    
    Args:
        name: Logger name (usually __name__)
        
    Returns:
        Logger instance
        
    Example:
        >>> from src.utils.logger import get_logger
        >>> logger = get_logger(__name__)
        >>> logger.info("Processing document")
    """
    return Logger.get_logger(name)


# Setup logger on module import
logger = Logger.setup()
