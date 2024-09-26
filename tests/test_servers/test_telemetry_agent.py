import time
import uuid

import pytest

import runhouse

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from runhouse.globals import rns_client
from runhouse.logger import get_logger
from runhouse.servers.telemetry.metrics_collection import (
    MetricsCollector,
    MetricsMetadata,
)
from runhouse.servers.telemetry.telemetry_agent import (
    ErrorCapturingExporter,
    TelemetryAgentReceiver,
)

logger = get_logger(__name__)


def provider_resource():
    return Resource.create(
        {"service.name": "runhouse-tests", "rh.version": runhouse.__version__}
    )


def metrics_metadata():
    return MetricsMetadata(username=rns_client.username, cluster_name="test")


def load_tracer():
    trace.set_tracer_provider(TracerProvider(resource=provider_resource()))
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
        provider = TracerProvider()
        trace.set_tracer_provider(provider)
        tracer = trace.get_tracer(__name__)

        endpoint = "grpc://telemetry.run.house"
        otlp_exporter = OTLPSpanExporter(
            endpoint=endpoint, headers=TelemetryAgentReceiver.request_headers()
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
    def test_send_interval_metrics_to_local_collector_backend(
        self, local_telemetry_collector
    ):
        """Generate cumulative metrics collection in-memory and send it to the local collector backend"""
        mc = MetricsCollector(
            metadata=metrics_metadata(),
            resource=provider_resource(),
            agent_endpoint="grpc://localhost:4316",
        )

        duration = 20
        start_time = time.time()
        while time.time() - start_time < duration:
            mc.update_cpu_utilization()
            time.sleep(5)

    @pytest.mark.level("local")
    def test_interval_metrics_to_runhouse_collector_backend(self):
        """Generate cumulative metrics collection in-memory and send it to the Runhouse collector backend"""
        endpoint = "grpc://telemetry.run.house:443"
        headers = TelemetryAgentReceiver.request_headers()

        mc = MetricsCollector(
            metadata=metrics_metadata(),
            resource=provider_resource(),
            agent_endpoint=endpoint,
            headers=headers,
        )

        duration = 20
        start_time = time.time()
        while time.time() - start_time < duration:
            mc.update_cpu_utilization()
            time.sleep(5)