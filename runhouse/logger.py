import logging
import os


def get_logger(name: str = __name__):
    logger = logging.getLogger(name)

    # Clear any existing handlers - avoid duplicate logs and maintain consistent log setup across modules
    logger.handlers.clear()

    level = os.getenv("RH_LOG_LEVEL")
    if level:
        try:
            logger.setLevel(getattr(logging, level.upper()))
        except AttributeError as e:
            raise e

    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        fmt="%(levelname)s | %(asctime)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger
