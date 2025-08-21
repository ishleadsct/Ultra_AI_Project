"""
Ultra AI Project - Logging Utilities

Logging configuration and utilities for the Ultra AI system.
"""

import logging
import sys
from pathlib import Path
from typing import Optional

def get_logger(name: str, level: str = "INFO") -> logging.Logger:
    """Get a configured logger instance."""
    logger = logging.getLogger(name)
    
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(getattr(logging, level.upper()))
    
    return logger

def setup_logging(log_file: Optional[Path] = None, level: str = "INFO"):
    """Set up logging configuration."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ] + ([logging.FileHandler(log_file)] if log_file else [])
    )
