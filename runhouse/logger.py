import logging
from io import StringIO


class StdoutHandler(logging.StreamHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stream = StringIO()

    def emit(self, record):
        try:
            msg = self.format(record)
            stream = self.stream
            stream.write(msg + self.terminator)
            self.flush()
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception:
            self.handleError(record)


def stdout_handler():
    """Factory to create and configure a custom StdoutHandler with StringIO as the stream. This allows for
    capturing logs as part of the stdout stream"""
    handler = StdoutHandler()
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(levelname)s | %(asctime)s | %(message)s")
    handler.setFormatter(formatter)
    handler.stream = StringIO()
    return handler


LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": True,
    "formatters": {
        "standard": {"format": "%(levelname)s | %(asctime)s | %(message)s"},
    },
    "handlers": {
        "stdout": {
            "()": "runhouse.logger.stdout_handler",
        },
    },
    "loggers": {
        "": {  # root logger
            "handlers": ["stdout"],
            "level": "INFO",
            "propagate": False,
        },
        "my.packg": {"handlers": ["stdout"], "level": "INFO", "propagate": False},
        "__main__": {  # if __name__ == '__main__'
            "handlers": ["stdout"],
            "level": "DEBUG",
            "propagate": False,
        },
    },
}
