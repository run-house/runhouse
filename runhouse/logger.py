import logging
import os


def get_logger(name: str = __name__):
    logger = logging.getLogger(name)

    level = os.getenv("RH_LOG_LEVEL")
    if level:
        try:
            logger.setLevel(getattr(logging, level.upper()))
        except AttributeError as e:
            raise e

    return logger
