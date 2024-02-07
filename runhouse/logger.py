import logging
from datetime import datetime, timezone
from typing import List


class ColoredFormatter:
    COLORS = {
        "black": "\u001b[30m",
        "red": "\u001b[31m",
        "green": "\u001b[32m",
        "yellow": "\u001b[33m",
        "blue": "\u001b[34m",
        "magenta": "\u001b[35m",
        "cyan": "\u001b[36m",
        "white": "\u001b[37m",
        "reset": "\u001b[0m",
    }

    @classmethod
    def get_color(cls, color: str):
        return cls.COLORS.get(color, "")


class FunctionLogHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.log_records = []

    def emit(self, record):
        self.log_records.append(record)

    @staticmethod
    def log_records_to_stdout(log_records: List[logging.LogRecord]) -> str:
        """Convert the log records to a string repr of the stdout output"""
        captured_logs = [
            f"{log_record.levelname} | {log_record.asctime} | {log_record.msg}"
            for log_record in log_records
        ]
        return "\n".join(captured_logs)


class UTCFormatter(logging.Formatter):
    """Ensure logs are always in UTC time"""

    @staticmethod
    def converter(timestamp):
        return datetime.fromtimestamp(timestamp, tz=timezone.utc)

    def formatTime(self, record, datefmt=None):
        dt = self.converter(record.created)
        if datefmt:
            return dt.strftime(datefmt)
        else:
            return dt.isoformat(timespec="milliseconds")


LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": True,
    "formatters": {
        "utc_formatter": {
            "()": UTCFormatter,
            "format": "%(levelname)s | %(asctime)s | %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S.%f",
        },
    },
    "handlers": {
        "default": {
            "level": "INFO",
            "formatter": "utc_formatter",
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stderr",  # Default is stderr
        },
    },
    "loggers": {
        "": {  # root logger
            "handlers": ["default"],
            "level": "INFO",
            "propagate": False,
        },
        "my.packg": {"handlers": ["default"], "level": "INFO", "propagate": False},
        "__main__": {  # if __name__ == '__main__'
            "handlers": ["default"],
            "level": "DEBUG",
            "propagate": False,
        },
    },
}
