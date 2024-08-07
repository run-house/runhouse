import logging
import time

from runhouse.constants import TELEMETRY_LOGS_URL

from runhouse.servers.telemetry.log_handler import RequestLogHandler

logger = logging.getLogger(__name__)


async def call_func_with_telemetry(func, *args, **kwargs):
    """Decorator to collect telemetry data for a function call."""
    from opentelemetry import trace

    from runhouse.servers.obj_store import ObjStore

    func_name = func.__name__

    # For logging internal function calls, we'll use a custom logger to gather more telemetry specific data
    # Save the original root logger handlers and level
    root_logger = logging.getLogger()
    original_handlers = root_logger.handlers[:]
    original_level = root_logger.level

    # Set up the custom logging configuration on the root logger
    custom_logger = logging.getLogger(__name__)
    log_handler = RequestLogHandler()
    custom_logger.addHandler(log_handler)
    custom_logger.setLevel(logging.INFO)

    # Apply this custom configuration to the root logger
    root_logger.handlers = custom_logger.handlers[:]
    root_logger.setLevel(custom_logger.level)

    tracer = trace.get_tracer(__name__)

    try:

        # Start span collection for this func execution
        with tracer.start_as_current_span(func_name) as span:
            # Extract cluster configuration
            cluster_config = (
                args[0].cluster_config if isinstance(args[0], ObjStore) else {}
            )

            # Extract function keyword arguments
            func_kwargs = kwargs.copy()
            func_kwargs.pop("data", None)

            # Enrich span with cluster_config data
            for key, value in cluster_config.items():
                span.set_attribute(f"rh.{key}", value)

            # Enrich span with func_kwargs data
            for key, value in func_kwargs.items():
                span.set_attribute(f"rh.{key}", value)

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

                # Unlike Traces and Metrics, there is no equivalent Logs API (only an SDK). Rather than turning the
                # Python logger into an OTLP logger we'll capture and send logs after function execution
                logs = log_handler.get_logs()
                if logs:
                    await send_logs_async(logs)

                log_handler.clear_logs()

    finally:
        # Restore the original root logger configuration
        root_logger.handlers = original_handlers
        root_logger.setLevel(original_level)

    return result


async def send_logs_async(logs):
    """Send logs data to the telemetry server."""
    import httpx

    async with httpx.AsyncClient() as client:
        response = await client.post(TELEMETRY_LOGS_URL, json={"logs": logs})
        if response.status_code != 200:
            logging.warning(
                f"({response.status_code}) Failed to send telemetry data: {response.text}"
            )
