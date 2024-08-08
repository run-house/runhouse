from typing import List

try:
    from opentelemetry import trace
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
    from opentelemetry.sdk.trace.export import (
        SimpleSpanProcessor,
        SpanExporter,
        SpanExportResult,
    )
except ImportError:
    pass


class InMemorySpanExporter(SpanExporter):
    def __init__(self):
        self.finished_spans = []

    def export(self, spans: List[ReadableSpan]) -> SpanExportResult:
        self.finished_spans.extend(spans)
        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:
        self.finished_spans.clear()

    def clear(self):
        self.finished_spans.clear()

    def get_finished_spans(self):
        return self.finished_spans


try:
    memory_exporter = InMemorySpanExporter()
except ImportError:
    memory_exporter = None
