import logging
import logging.config
import os

from runhouse.constants import DEFAULT_LOG_LEVEL


def get_logger(name: str = __name__):
    logger = logging.getLogger(name)
    level = os.getenv("RH_LOG_LEVEL", DEFAULT_LOG_LEVEL).upper()

    if level:
        # Set the logging level
        logger.setLevel(level)

    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            fmt="%(levelname)s | %(asctime)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    # Prevent the logger from propagating to the root logger
    logger.propagate = False

    return logger
