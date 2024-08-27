import logging
import os
import sys


def get_logger(reinitialize: bool = False) -> logging.Logger:
    """
    Creates and returns a logger with the specified name.

    Ensures a universal logger configuration across the codebase with the format:
    "levelname - asctime - filename:lineno - message"

    Args:
        name (str): Name of the logger. Defaults to None, which gets the root logger.

    Returns:
        logging.Logger: Configured logger instance.
    """
    # Create or retrieve the logger
    logger = logging.getLogger("_rh_universal_logger")

    level = os.getenv("RH_LOG_LEVEL") or "INFO"
    try:
        level = getattr(logging, level.upper())
    except AttributeError as e:
        raise e

    if reinitialize:
        for handler in logger.handlers:
            logger.removeHandler(handler)

    # Check if the logger has already been configured to prevent duplicate handlers
    if not logger.handlers:

        # Set the level for the new logger
        logger.setLevel(level)

        # Create a console handler that outputs to stdout
        console_handler = logging.StreamHandler(stream=sys.stdout)

        # The handler should capture all log levels, so we set the level to debug
        console_handler.setLevel(logging.DEBUG)

        # Define the log format
        formatter = logging.Formatter(
            "%(levelname)s | %(asctime)s | %(filename)s:%(lineno)d | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        # Apply the formatter to the handler
        console_handler.setFormatter(formatter)

        # Add the handler to the logger
        logger.addHandler(console_handler)

        # Prevent log messages from propagating to the root logger
        logger.propagate = False

    return logger
