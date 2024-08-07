import logging
import time

import requests

from runhouse.servers.telemetry import memory_exporter
from runhouse.servers.telemetry.log_handler import RequestLogHandler


async def call_func_with_telemetry(func, *args, **kwargs):
    """Decorator to collect telemetry data for a function call."""
    from opentelemetry import trace

    func_name = func.__name__
    handlers = logging.getLogger(__name__).handlers
    formatter = handlers[0].formatter if handlers else None

    log_handler = RequestLogHandler(formatter=formatter)
    logging.getLogger(__name__).addHandler(log_handler)

    tracer = trace.get_tracer(__name__)
    with tracer.start_as_current_span(func_name) as span:
        start_time = time.time()

        try:
            result = await func(*args, **kwargs)
            span.set_attribute("status", "success")

        except Exception as e:
            span.set_attribute("error", str(e))
            span.set_attribute("status", "error")
            raise e

        finally:
            end_time = time.time()
            duration = end_time - start_time

            span.set_attribute("duration", duration)
            span.end()

            # Collect logs
            logs = log_handler.get_logs()
            logging.getLogger(__name__).removeHandler(log_handler)

            # Collect spans
            spans = memory_exporter.get_finished_spans()
            formatted_spans = [s.to_json() for s in spans]

            # TODO: save other obj store metadata? (ex: imported modules, installed envs, etc.)
            cluster_config = args[0].cluster_config if args else {}

            func_kwargs = kwargs.copy()
            func_kwargs.pop("data", None)

            resp = requests.post(
                "https://telemetry.run.house/api/v1/telemetry",
                json={
                    "spans": formatted_spans,
                    "logs": logs,
                    "resource_metadata": {**func_kwargs, **cluster_config},
                },
            )
            if resp.status_code != 200:
                logging.debug(
                    f"({resp.status_code}) Failed to send telemetry data: {resp.text}"
                )

            # Clear the in-memory exporter after sending
            memory_exporter.clear()

    return result
