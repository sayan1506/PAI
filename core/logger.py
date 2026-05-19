"""
PAI Logging System

Configures Loguru with two handlers:
- Console handler: Colorized output to stderr at configured LOG_LEVEL
- File handler: Writes to logs/pai.log with 5MB rotation, 7-day retention, zip compression

Usage:
    from core.logger import logger
"""

import os
import sys
from loguru import logger

import config

# Ensure logs directory exists before adding file handler
os.makedirs("logs", exist_ok=True)

# Remove default Loguru handler
logger.remove()

# Console handler — colorized, readable, respects configured LOG_LEVEL
logger.add(
    sys.stderr,
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan> - <level>{message}</level>",
    level=config.LOG_LEVEL,
    colorize=True,
)

# File handler — full details, rotates at 5MB, retains 7 days, compresses old logs
logger.add(
    "logs/pai.log",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
    level="DEBUG",
    rotation="5 MB",
    retention="7 days",
    compression="zip",
)

__all__ = ["logger"]
