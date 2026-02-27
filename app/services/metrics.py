from __future__ import annotations

from dataclasses import dataclass

from prometheus_client import Counter, Histogram


REQUEST_COUNTER = Counter("photo_editor_requests_total", "Total generation requests")
REJECT_COUNTER = Counter("photo_editor_rejections_total", "Total rejected generations")
LATENCY_HIST = Histogram("photo_editor_latency_seconds", "Generation latency", buckets=(0.5, 1, 2, 4, 8, 12, 20))
IDENTITY_SIM_HIST = Histogram(
    "photo_editor_identity_similarity",
    "Identity similarity distribution",
    buckets=(0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0),
)


@dataclass
class MetricsService:
    def mark_request(self) -> None:
        REQUEST_COUNTER.inc()

    def mark_rejection(self) -> None:
        REJECT_COUNTER.inc()

    def observe_latency(self, value_seconds: float) -> None:
        LATENCY_HIST.observe(value_seconds)

    def observe_identity_similarity(self, value: float) -> None:
        IDENTITY_SIM_HIST.observe(value)
