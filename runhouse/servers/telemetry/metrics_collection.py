import GPUtil
import psutil

from opentelemetry import metrics
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics._internal.measurement import Measurement
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader

from runhouse.constants import METRICS_EXPORT_INTERVAL_MS
from runhouse.logger import get_logger

logger = get_logger(__name__)


class MetricsCollector:
    def __init__(
        self,
        agent_endpoint: str,
        resource_attributes,
        headers: dict = None,
    ):
        from runhouse.servers.telemetry import TelemetryAgentReceiver

        self._instrument_cache = {}
        self.endpoint = agent_endpoint
        self.headers = headers

        self.resource = TelemetryAgentReceiver.default_resource(resource_attributes)
        self.meter = self._load_metrics_meter()

    def _load_metrics_meter(self):
        """Load the metrics meter with the OTLP exporter. If headers are provided, use them for authentication."""
        metrics_exporter = (
            OTLPMetricExporter(endpoint=self.endpoint, headers=self.headers)
            if self.headers
            else OTLPMetricExporter(endpoint=self.endpoint, insecure=True)
        )

        # Collect metrics based on a configured time interval
        metric_reader = PeriodicExportingMetricReader(
            metrics_exporter, export_interval_millis=METRICS_EXPORT_INTERVAL_MS
        )
        provider = MeterProvider(metric_readers=[metric_reader], resource=self.resource)

        metrics.set_meter_provider(provider)
        return metrics.get_meter(__name__)

    @staticmethod
    def get_gpu_metadata():
        gpus = GPUtil.getGPUs()
        total_gpu_memory = sum(gpu.memoryTotal for gpu in gpus) if gpus else 0
        total_used_memory = sum(gpu.memoryUsed for gpu in gpus) if gpus else 0
        free_memory = total_gpu_memory - total_used_memory if total_gpu_memory else 0
        gpu_utilization_percent = (
            sum(gpu.load * 100 for gpu in gpus) / len(gpus) if gpus else 0
        )

        return {
            "total_memory": total_gpu_memory,
            "used_memory": total_used_memory,
            "free_memory": free_memory,
            "gpu_count": len(gpus),
            "utilization_percent": round(gpu_utilization_percent, 2),
        }

    @staticmethod
    def get_cpu_metadata():
        cpu_percent = psutil.cpu_percent()
        virtual_memory = psutil.virtual_memory()

        # Convert to MB
        total_cpu_memory = virtual_memory.total / (1024**2)
        used_cpu_memory = virtual_memory.used / (1024**2)
        free_cpu_memory = virtual_memory.free / (1024**2)

        return {
            "utilization_percent": cpu_percent,
            "total_memory": total_cpu_memory,
            "used_memory": used_cpu_memory,
            "free_memory": free_cpu_memory,
        }

    def update_cpu_metrics(self):
        cpu_utilization_counter = self._get_or_create_counter(
            "cpu_utilization_total", "Total CPU utilization over time", "percentage"
        )
        cpu_usage = self.get_cpu_metadata()
        cpu_utilization_counter.add(
            cpu_usage["utilization_percent"],
            {
                "unit": "percentage",
            },
        )

        # Record used memory
        cpu_memory_usage_histogram = self._get_or_create_histogram(
            "cpu_memory_usage", "CPU memory usage", "MB"
        )
        cpu_memory_usage_histogram.record(
            cpu_usage["used_memory"],
            {
                "unit": "MB",
            },
        )

        # Record free memory
        cpu_free_memory_histogram = self._get_or_create_histogram(
            "cpu_free_memory", "CPU free memory", "MB"
        )
        cpu_free_memory_histogram.record(
            cpu_usage["free_memory"],
            {
                "unit": "MB",
            },
        )

    def update_gpu_metrics(self):
        gpu_utilization_counter = self._get_or_create_counter(
            "gpu_utilization_total", "Total GPU utilization over time", "percentage"
        )
        gpu_usage = self.get_gpu_metadata()
        gpu_utilization_counter.add(
            gpu_usage["utilization_percent"],
            {
                "unit": "percentage",
            },
        )

        # Record used memory
        gpu_memory_usage_histogram = self._get_or_create_histogram(
            "cpu_memory_usage", "GPU memory usage", "MB"
        )
        gpu_memory_usage_histogram.record(
            gpu_usage["used_memory"],
            {
                "unit": "MB",
            },
        )

        # Record free memory
        gpu_free_memory_histogram = self._get_or_create_histogram(
            "cpu_free_memory", "GPU free memory", "MB"
        )
        gpu_free_memory_histogram.record(
            gpu_usage["free_memory"],
            {
                "unit": "MB",
            },
        )

        # Record GPU count
        gpu_count_histogram = self._get_or_create_histogram(
            "gpu_count", "Number of GPUs", "count"
        )
        gpu_count_histogram.record(
            gpu_usage["gpu_count"],
            {
                "unit": "count",
            },
        )

    def cpu_memory_usage_callback(self, instrument, options):
        cpu_usage = self.get_cpu_metadata()
        return [
            Measurement(
                instrument=instrument,
                value=cpu_usage["used_memory"],
                attributes={
                    "unit": "MB",
                },
            )
        ]

    def cpu_free_memory_callback(self, instrument, options):
        cpu_usage = self.get_cpu_metadata()
        return [
            Measurement(
                instrument=instrument,
                value=cpu_usage["free_memory"],
                attributes={
                    "unit": "MB",
                },
            )
        ]

    def gpu_memory_usage_callback(self, instrument, options):
        gpu_usage = self.get_gpu_metadata()
        return [
            Measurement(
                instrument=instrument,
                value=gpu_usage["used_memory"],
                attributes={
                    "unit": "MB",
                },
            )
        ]

    def gpu_count_callback(self, instrument, options):
        gpu_usage = self.get_gpu_metadata()
        return [
            Measurement(
                instrument=instrument,
                value=gpu_usage["gpu_count"],
                attributes={
                    "unit": "count",
                },
            )
        ]

    def initialize_cpu_metrics_counter(self):
        """Initialize the metrics counters and observable gauges for CPU utilization."""
        self._get_or_create_observable_gauge(
            "cpu_memory_usage",
            self.cpu_memory_usage_callback,
            "CPU memory usage in MB",
            "MB",
        )

        self._get_or_create_observable_gauge(
            "cpu_free_memory",
            self.cpu_free_memory_callback,
            "Free CPU memory in MB",
            "MB",
        )

    def initialize_gpu_metrics_counter(self):
        """Initialize the metrics counters and observable gauges for GPU utilization."""
        self._get_or_create_observable_gauge(
            "gpu_memory_usage",
            self.gpu_memory_usage_callback,
            "Total GPU memory usage",
            "MB",
        )

        self._get_or_create_observable_gauge(
            "gpu_count", self.gpu_count_callback, "Number of GPUs", "count"
        )

    ##################################################################
    # Helper Methods to manage instruments
    ##################################################################

    def _get_or_create_counter(self, name, description, unit):
        """Retrieve or create a Counter instrument."""
        if name not in self._instrument_cache:
            self._instrument_cache[name] = self.meter.create_counter(
                name, description=description, unit=unit
            )
        return self._instrument_cache[name]

    def _get_or_create_observable_gauge(self, name, callback, description, unit):
        """Retrieve or create an ObservableGauge instrument."""
        if name not in self._instrument_cache:
            self._instrument_cache[name] = self.meter.create_observable_gauge(
                name, callbacks=[callback], description=description, unit=unit
            )
        return self._instrument_cache[name]

    def _get_or_create_histogram(self, name, description, unit):
        """Retrieve or create a Histogram instrument."""
        if name not in self._instrument_cache:
            self._instrument_cache[name] = self.meter.create_histogram(
                name, description=description, unit=unit
            )
        return self._instrument_cache[name]
