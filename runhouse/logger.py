import logging
import os
import sys


def get_logger(name: str = __name__):
    logger = logging.getLogger(name)

    level = os.getenv("RH_LOG_LEVEL")
    if level:
        # Set the logging level
        logger.setLevel(level.upper())

    # Check if the logger already has handlers, add a StreamHandler if not
    if not logger.handlers:
        # Use sys.stdout managed by StreamTee
        handler = logging.StreamHandler(stream=sys.stdout)
        formatter = logging.Formatter(
            fmt="%(levelname)s | %(asctime)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    # Prevent the logger from propagating to the root logger
    logger.propagate = False

    return logger
