import logging
import logging.config
import re
from datetime import datetime, timezone
from typing import List, Union

from runhouse.constants import DEFAULT_LOG_LEVEL


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

    # TODO: This method is a temp solution, until we'll update logging architecture. Remove once logging is cleaned up.
    @classmethod
    def format_log(cls, text):
        ansi_escape = re.compile(r"(?:\x1B[@-_][0-?]*[ -/]*[@-~])")
        return ansi_escape.sub("", text)


class ClusterLogsFormatter:
    def __init__(self, system):
        self.system = system
        self._display_title = False

    def format(self, output_type):
        from runhouse import Resource
        from runhouse.servers.http.http_utils import OutputType

        system_color = ColoredFormatter.get_color("cyan")
        reset_color = ColoredFormatter.get_color("reset")

        prettify_logs = output_type in [
            OutputType.STDOUT,
            OutputType.EXCEPTION,
            OutputType.STDERR,
        ]

        if (
            isinstance(self.system, Resource)
            and prettify_logs
            and not self._display_title
        ):
            # Display the system name before subsequent logs only once
            system_name = self.system.name
            dotted_line = "-" * len(system_name)
            print(dotted_line)
            print(f"{system_color}{system_name}{reset_color}")
            print(dotted_line)

            # Only display the system name once
            self._display_title = True

        return system_color, reset_color


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


def get_logger(name: str = __name__, log_level: Union[int, str] = logging.INFO):
    level_name = (
        logging.getLevelName(log_level) if isinstance(log_level, int) else log_level
    )
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
                "level": level_name,
                "formatter": "utc_formatter",
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stderr",  # Default is stderr
            },
        },
        "loggers": {
            "": {  # root logger
                "handlers": ["default"],
                "level": level_name,
                "propagate": False,
            },
            "my.packg": {
                "handlers": ["default"],
                "level": level_name,
                "propagate": False,
            },
            "__main__": {  # if __name__ == '__main__'
                "handlers": ["default"],
                "level": level_name,
                "propagate": False,
            },
        },
    }
    logging.config.dictConfig(LOGGING_CONFIG)
    logger = logging.getLogger(name=name)
    return logger


logger = get_logger(name=__name__, log_level=DEFAULT_LOG_LEVEL)
