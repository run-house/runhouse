import logging
from typing import List


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


LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": True,
    "formatters": {
        "standard": {"format": "%(levelname)s | %(asctime)s | %(message)s"},
    },
    "handlers": {
        "default": {
            "level": "INFO",
            "formatter": "standard",
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
