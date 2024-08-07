import logging


class RequestLogHandler(logging.Handler):
    def __init__(self, formatter=None):
        super().__init__()
        self.logs = []
        self.setFormatter(formatter)

    def emit(self, record):
        log_entry = {
            "level": record.levelname,
            "message": self.format(record),
        }
        self.logs.append(log_entry)

    def get_logs(self):
        return self.logs

    def clear_logs(self):
        self.logs = []
