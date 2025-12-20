"""
Module: logging_utils
=====================

Provides shared utilities for logging configuration across the project.
It ensures a consistent log format and handles file/console output switching.

Functions:
    - setup_logger_context: A context manager for scoped logging.
"""

import logging
from contextlib import contextmanager

# Shared formatter to ensure consistent timestamp and level display across all logs
SHARED_FORMATTER = logging.Formatter('%(asctime)s - [%(levelname)s] - %(message)s')


@contextmanager
def setup_logger_context(name: str, log_file_path: str = None, console: bool = False):
    """
    Context Manager that sets up a logger configuration for a specific scope.

    It attaches a FileHandler (and optionally a StreamHandler for console output)
    to a logger, yields the logger for use, and ensures proper cleanup
    (closing files, removing handlers) afterwards.

    Args:
        name (str): The unique name of the logger (e.g., "Global" or "Local_hash").
        log_file_path (str or None): Full path to the log file. If None, no file is written.
        console (bool): If True, also outputs logs to the standard console (stdout).

    Yields:
        logging.Logger: The configured logger instance.
    """
    # Retrieve or create the logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Keep track of handlers created in this context to close/remove them later
    active_handlers = []

    # --- FILE HANDLER ---
    if log_file_path:
        file_handler = logging.FileHandler(log_file_path, mode='a')
        file_handler.setFormatter(SHARED_FORMATTER)
        logger.addHandler(file_handler)
        active_handlers.append(file_handler)

    # --- CONSOLE HANDLER ---
    if console:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(SHARED_FORMATTER)
        logger.addHandler(console_handler)
        active_handlers.append(console_handler)

    # Prevent logs from bubbling up to the root logger (duplicates)
    logger.propagate = False

    try:
        yield logger

    finally:
        # Clean up resources
        for handler in active_handlers:
            handler.close()
            logger.removeHandler(handler)
