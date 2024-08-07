import logging


class RequestLogHandler(logging.Handler):
    """Custom logger to use when capturing function and module execution on the cluster. Captures
    additional telemetry data to enrich the logs."""

    def __init__(self, formatter=None):
        super().__init__()
        self.logs = []
        self.setFormatter(formatter)

    def emit(self, record):
        from opentelemetry import trace

        # Add the span and trace ID to the log record where relevant
        span = trace.get_current_span()

        if span and span.get_span_context().is_valid:
            record.trace_id = format(span.get_span_context().trace_id, "032x")
            record.span_id = format(span.get_span_context().span_id, "016x")
        else:
            record.trace_id = None
            record.span_id = None

        self.logs.append(
            {
                "body": record.msg,
                "severity_text": record.levelname,
                "severity_number": record.levelno,
                "trace_id": str(record.trace_id),
                "span_id": str(record.span_id),
                "created": record.created,
            }
        )

    def get_logs(self):
        return self.logs

    def clear_logs(self):
        self.logs = []
