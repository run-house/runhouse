import time
import uuid

import psutil
import pytest

from opentelemetry import metrics, trace
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics._internal.measurement import Measurement
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from runhouse.logger import get_logger

logger = get_logger(__name__)


def current_time():
    return int(time.time() * 1e9)


def load_tracer():
    resource = Resource.create({"service.name": "runhouse-tests"})
    trace.set_tracer_provider(TracerProvider(resource=resource))
    tracer = trace.get_tracer(__name__)
    return tracer


# Function to get CPU utilization
def get_cpu_utilization(instrument):
    # TODO: use Runhouse custom collection methods
    return [Measurement(instrument=instrument, value=psutil.cpu_percent())]


# Function to get GPU utilization
def get_gpu_utilization(instrument):
    # TODO: use Runhouse custom collection methods
    import GPUtil

    gpus = GPUtil.getGPUs()
    if gpus:
        return [Measurement(instrument=instrument, value=gpus[0].load * 100)]
    return [Measurement(instrument=instrument, value=0)]


def get_gpu_memory_usage(instrument):
    # TODO: use Runhouse custom collection methods
    import GPUtil

    gpus = GPUtil.getGPUs()
    if gpus:
        return [
            Measurement(instrument=instrument, value=gpus[0].memoryUtil * 100)
        ]  # GPU memory usage is a fraction
    return [Measurement(instrument=instrument, value=0)]


def update_cpu_utilization(counter):
    # Function to add CPU utilization to the counter
    cpu_percent = psutil.cpu_percent()
    counter.add(cpu_percent, {"unit": "percentage"})


def update_gpu_utilization(counter):
    # Function to add GPU utilization to the counter
    import GPUtil

    gpus = GPUtil.getGPUs()
    if gpus:
        counter.add(gpus[0].load * 100, {"unit": "percentage"})
    else:
        counter.add(0, {"unit": "percentage"})


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
        """Generate a span and have a locally running Otel agent send it to a locally running collector backend"""
        provider = TracerProvider()
        trace.set_tracer_provider(provider)
        tracer = trace.get_tracer(__name__)

        # Have the agent be responsible for sending the spans to the collector backend
        endpoint = f"grpc://localhost:{local_telemetry_agent_for_local_backend.agent_config.grpc_port}"
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
        endpoint = f"grpc://localhost:{local_telemetry_agent_for_runhouse_backend.agent_config.grpc_port}"
        span_processor = BatchSpanProcessor(
            OTLPSpanExporter(endpoint=endpoint, insecure=True)
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
        endpoint = "grpc://telemetry.run.house"
        span_processor = BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint))
        trace.get_tracer_provider().add_span_processor(span_processor)

        with tracer.start_as_current_span(f"span-from-agent-{str(uuid.uuid4())}"):
            logger.info("Test span created and sent to the collector by the agent!")

        # Force flush of the span processor
        provider.force_flush()
        with tracer.start_as_current_span("span-from-local"):
            logger.info("Test span created and sent to the collector!")

    @pytest.mark.level("local")
    def test_send_current_metrics_to_runhouse_collector_backend(
        self, local_telemetry_collector
    ):
        """Generate metrics collection in-memory and send it to the local collector backend"""
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

        # Create instruments to collect CPU and GPU metrics at specific points in time
        cpu_utilization_gauge = meter.create_observable_gauge(
            "cpu_utilization",
            callbacks=[lambda options: get_cpu_utilization(cpu_utilization_gauge)],
            description="System CPU Utilization",
            unit="percentage",
        )

        gpu_utilization_gauge = meter.create_observable_gauge(
            "gpu_utilization",
            callbacks=[lambda options: get_gpu_utilization(gpu_utilization_gauge)],
            description="GPU Utilization",
            unit="percentage",
        )

        gpu_memory_gauge = meter.create_observable_gauge(
            "gpu_memory_usage",
            callbacks=[lambda options: get_gpu_memory_usage(gpu_memory_gauge)],
            description="GPU Memory Usage",
            unit="percentage",
        )

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

        gpu_utilization_counter = meter.create_counter(
            "gpu_utilization_total",
            description="Total GPU utilization over time",
            unit="percentage",
        )

        duration = 20  # Duration to run the collection in seconds
        start_time = time.time()
        while time.time() - start_time < duration:
            update_cpu_utilization(cpu_utilization_counter)
            update_gpu_utilization(gpu_utilization_counter)

            # Collect data every 5 seconds
            time.sleep(5)
