import uuid

import pytest

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from runhouse.logger import get_logger

logger = get_logger(__name__)


@pytest.mark.servertest
class TestTelemetryAgent:
    """Sets up a locally running telemetry agent instance, and sends off the telemetry data to a locally running
    collector backend (mocking the Runhouse collector backend)."""

    @pytest.mark.level("local")
<<<<<<< HEAD

    def test_send_span_to_collector_backend(self, local_telemetry_collector):
        """Generate a span and send it to a locally running collector backend"""

=======
    def test_send_span_to_collector_backend(self, local_telemetry_collector):
        """Generate a span locally in-memory and send it to a locally running collector backend"""
>>>>>>> ae221199 (instrument obj store method with otel agent)
        trace.set_tracer_provider(TracerProvider())
        tracer = trace.get_tracer(__name__)

        # Send spans directly to the collector backend without an agent process
        span_processor = BatchSpanProcessor(
            OTLPSpanExporter(
                endpoint="grpc://localhost:4316",
                insecure=True,
            )
        )
        trace.get_tracer_provider().add_span_processor(span_processor)

        with tracer.start_as_current_span(f"test-span-{str(uuid.uuid4())}"):
            logger.info("Test span created and sent to the collector!")

    @pytest.mark.level("local")
    def test_send_span_with_local_agent_to_local_collector_backend(
        self, local_telemetry_collector, local_telemetry_agent_for_local_backend
    ):
<<<<<<< HEAD
        """Generate a span and have a locally running Otel agent send it to a locally running collector backend"""
=======
        """Generate a span and have a local Otel agent send it to a locally running collector backend"""
>>>>>>> ae221199 (instrument obj store method with otel agent)
        provider = TracerProvider()
        trace.set_tracer_provider(provider)
        tracer = trace.get_tracer(__name__)

        # Have the agent be responsible for sending the spans to the collector backend
<<<<<<< HEAD
        endpoint = f"grpc://localhost:{local_telemetry_agent_for_local_backend.agent_config.grpc_port}"
=======
        endpoint = f"grpc://localhost:{local_telemetry_agent_for_local_backend.config.grpc_port}"
>>>>>>> ae221199 (instrument obj store method with otel agent)
        span_processor = BatchSpanProcessor(
            OTLPSpanExporter(
                endpoint=endpoint,
                insecure=True,
                timeout=10,
            )
        )
        trace.get_tracer_provider().add_span_processor(span_processor)

        with tracer.start_as_current_span(f"span-from-agent-{str(uuid.uuid4())}"):
            logger.info("Test span created and sent to the collector by the agent!")

        # Force flush of the span processor
        provider.force_flush()

    @pytest.mark.level("local")
    def test_send_span_with_local_agent_to_collector_backend(
        self, local_telemetry_agent_for_runhouse_backend
    ):
        """Generate a span and have a local Otel agent send it to the Runhouse collector backend"""
        provider = TracerProvider()
        trace.set_tracer_provider(provider)
        tracer = trace.get_tracer(__name__)

        # Have the agent be responsible for sending the spans to the collector backend
<<<<<<< HEAD
        endpoint = f"grpc://localhost:{local_telemetry_agent_for_runhouse_backend.agent_config.grpc_port}"
        span_processor = BatchSpanProcessor(
            OTLPSpanExporter(endpoint=endpoint, insecure=True)
=======
        endpoint = f"grpc://localhost:{local_telemetry_agent_for_runhouse_backend.config.grpc_port}"
        span_processor = BatchSpanProcessor(
            OTLPSpanExporter(
                endpoint=endpoint,
                insecure=True,
                timeout=10,
            )
>>>>>>> ae221199 (instrument obj store method with otel agent)
        )
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

        # Have the agent be responsible for sending the spans to the collector backend
<<<<<<< HEAD
        endpoint = "grpc://telemetry.run.house"
        span_processor = BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint))
=======
        endpoint = "grpc://telemetry.run.house:4317"
        span_processor = BatchSpanProcessor(
            OTLPSpanExporter(
                endpoint=endpoint,
                # insecure=True,
                # timeout=10,
            )
        )
>>>>>>> ae221199 (instrument obj store method with otel agent)
        trace.get_tracer_provider().add_span_processor(span_processor)

        with tracer.start_as_current_span(f"span-from-agent-{str(uuid.uuid4())}"):
            logger.info("Test span created and sent to the collector by the agent!")

        # Force flush of the span processor
        provider.force_flush()
