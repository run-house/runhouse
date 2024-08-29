import logging
import os


def get_logger(name) -> logging.Logger:
    """
    Creates and returns a logger with the specified name.

    Ensures a universal logger configuration across the codebase with the format:
    "levelname - asctime - name:lineno - message"

    Args:
        name (str): Name of the logger. Defaults to None, which gets the root logger.

    Returns:
        logging.Logger: Configured logger instance.
    """
    # Create or retrieve the logger
    logger = logging.getLogger(name)

    logger.handlers.clear()  # Clear existing handlers
    level = os.getenv("RH_LOG_LEVEL") or "INFO"
    level = getattr(logging, level.upper())
    logger.setLevel(level)
    formatter = logging.Formatter(
        "%(levelname)s | %(asctime)s | %(name)s:%(lineno)d | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger
