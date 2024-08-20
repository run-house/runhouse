import logging
import os


def get_logger():
    logger = logging.getLogger(__name__)

    level = os.getenv("RH_LOG_LEVEL")
    if level:
        # Set the logging level
        logger.setLevel(level.upper())

    # Apply a custom formatter
    formatter = logging.Formatter(
        fmt="%(levelname)s | %(asctime)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Apply the formatter to each handler
    for handler in logger.handlers:
        handler.setFormatter(formatter)

    # Prevent the logger from propagating to the root logger
    logger.propagate = False

    return logger
