import time
import uuid

import pytest

from opentelemetry import metrics, trace
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from runhouse.logger import get_logger
from runhouse.servers.telemetry.metrics_collection import (
    update_cpu_utilization,
    update_gpu_utilization,
)
from runhouse.servers.telemetry.telemetry_agent import ErrorCapturingExporter

logger = get_logger(__name__)


def current_time():
    return int(time.time() * 1e9)


def load_tracer():
    resource = Resource.create({"service.name": "runhouse-tests"})
    trace.set_tracer_provider(TracerProvider(resource=resource))
    tracer = trace.get_tracer(__name__)
    return tracer


@pytest.mark.servertest
class TestTelemetryAgent:
    """Sets up a locally running telemetry agent instance, and sends off the telemetry data to a locally running
    collector backend (mocking the Runhouse collector backend)."""

    @pytest.mark.level("local")
    def test_send_span_to_collector_backend(self, local_telemetry_collector):
        """Generate a span locally in-memory and send it to a locally running collector backend"""
        trace.set_tracer_provider(TracerProvider())
        tracer = trace.get_tracer(__name__)

        # Send spans directly to the collector backend without an agent process
        endpoint = "grpc://localhost:4316"
        otlp_exporter = OTLPSpanExporter(endpoint=endpoint, insecure=True)
        error_capturing_exporter = ErrorCapturingExporter(otlp_exporter)
        span_processor = BatchSpanProcessor(error_capturing_exporter)
        trace.get_tracer_provider().add_span_processor(span_processor)

        with tracer.start_as_current_span(f"test-span-{str(uuid.uuid4())}"):
            logger.info("Test span created and sent to the collector!")

        assert (
            not error_capturing_exporter.has_errors()
        ), error_capturing_exporter.get_errors()

    @pytest.mark.level("local")
    def test_send_span_with_local_agent_to_local_collector_backend(
        self, local_telemetry_collector, local_telemetry_agent_for_local_backend
    ):
        """Generate a span and have a locally running Otel agent send it to a locally running collector backend"""
        provider = TracerProvider()
        trace.set_tracer_provider(provider)
        tracer = trace.get_tracer(__name__)

        # Have the agent be responsible for sending the spans to the collector backend
        endpoint = f"grpc://localhost:{local_telemetry_agent_for_local_backend.agent_config.grpc_port}"
        otlp_exporter = OTLPSpanExporter(endpoint=endpoint, insecure=True)
        error_capturing_exporter = ErrorCapturingExporter(otlp_exporter)
        span_processor = BatchSpanProcessor(error_capturing_exporter)
        trace.get_tracer_provider().add_span_processor(span_processor)

        with tracer.start_as_current_span(f"span-from-agent-{str(uuid.uuid4())}"):
            logger.info("Test span created and sent to the collector by the agent!")

        # Force flush of the span processor
        provider.force_flush()

        assert (
            not error_capturing_exporter.has_errors()
        ), error_capturing_exporter.get_errors()

    @pytest.mark.level("local")
    def test_send_span_with_local_agent_to_collector_backend(
        self, local_telemetry_agent_for_runhouse_backend
    ):
        """Generate a span and have a local Otel agent send it to the Runhouse collector backend"""
        provider = TracerProvider()
        trace.set_tracer_provider(provider)
        tracer = trace.get_tracer(__name__)

        # Have the agent be responsible for sending the spans to the collector backend
        endpoint = f"grpc://localhost:{local_telemetry_agent_for_runhouse_backend.agent_config.grpc_port}"
        otlp_exporter = OTLPSpanExporter(endpoint=endpoint, insecure=True)
        error_capturing_exporter = ErrorCapturingExporter(otlp_exporter)
        span_processor = BatchSpanProcessor(error_capturing_exporter)
        trace.get_tracer_provider().add_span_processor(span_processor)

        with tracer.start_as_current_span(f"span-from-agent-{str(uuid.uuid4())}"):
            logger.info("Test span created and sent to the collector by the agent!")

        # Force flush of the span processor
        provider.force_flush()

    @pytest.mark.level("local")
    def test_send_span_to_runhouse_collector_backend(self):
        """Generate a span in-memory and send it to the Runhouse collector backend"""
        from runhouse.servers.telemetry import TelemetryAgentExporter

        provider = TracerProvider()
        trace.set_tracer_provider(provider)
        tracer = trace.get_tracer(__name__)

        endpoint = "grpc://telemetry.run.house"
        otlp_exporter = OTLPSpanExporter(
            endpoint=endpoint, headers=TelemetryAgentExporter.request_headers()
        )
        error_capturing_exporter = ErrorCapturingExporter(otlp_exporter)
        span_processor = BatchSpanProcessor(error_capturing_exporter)
        trace.get_tracer_provider().add_span_processor(span_processor)

        with tracer.start_as_current_span(f"span-from-agent-{str(uuid.uuid4())}"):
            logger.info("Test span created and sent to the collector by the agent!")

        # Force flush of the span processor
        provider.force_flush()

        # Assert that no errors occurred, otherwise log all error messages
        assert (
            not error_capturing_exporter.has_errors()
        ), error_capturing_exporter.get_errors()

    @pytest.mark.level("local")
    def test_cumulative_current_metrics_to_runhouse_collector_backend(
        self, local_telemetry_collector
    ):
        """Generate cumulative metrics collection in-memory and send it to the local collector backend"""

        # Set up the OTLP exporter
        exporter = OTLPMetricExporter(endpoint="grpc://localhost:4316", insecure=True)

        # collect metrics based on a user-configurable time interval, and pass the metrics to the exporter
        metric_reader = PeriodicExportingMetricReader(
            exporter, export_interval_millis=5000
        )
        provider = MeterProvider(metric_readers=[metric_reader])

        # Set the global MeterProvider
        metrics.set_meter_provider(provider)
        meter = metrics.get_meter(__name__)

        # Create counters to accumulate CPU and GPU utilization
        cpu_utilization_counter = meter.create_counter(
            "cpu_utilization_total",
            description="Total CPU utilization over time",
            unit="percentage",
        )

        cpu_memory_usage_gauge = meter.create_observable_gauge(
            "cpu_memory_usage",
            description="CPU memory usage in MB",
            unit="MB",
        )

        cpu_free_memory_gauge = meter.create_observable_gauge(
            "cpu_free_memory",
            description="Free CPU memory in MB",
            unit="MB",
        )

        gpu_memory_usage_gauge = meter.create_observable_gauge(
            "gpu_memory_usage",
            description="Total GPU memory usage",
            unit="MB",
        )

        gpu_utilization_counter = meter.create_counter(
            "gpu_utilization_total",
            description="Total GPU utilization over time",
            unit="percentage",
        )

        gpu_count_gauge = meter.create_observable_gauge(
            "gpu_count",
            description="Number of GPUs",
            unit="count",
        )

        duration = 20  # Duration to run the collection in seconds
        start_time = time.time()
        while time.time() - start_time < duration:
            update_cpu_utilization(
                cpu_utilization_counter, cpu_memory_usage_gauge, cpu_free_memory_gauge
            )
            update_gpu_utilization(
                gpu_utilization_counter, gpu_memory_usage_gauge, gpu_count_gauge
            )

            # Collect data every 5 seconds
            time.sleep(5)
