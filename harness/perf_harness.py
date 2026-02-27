from __future__ import annotations

import time
from dataclasses import dataclass

from app.domain.schemas import GenerationRequest
from app.services.orchestrator import OrchestratorService


@dataclass
class PerfSummary:
    runs: int
    p95_latency_ms: float
    p99_latency_ms: float


class PerformanceHarness:
    def __init__(self, orchestrator: OrchestratorService) -> None:
        self.orchestrator = orchestrator

    def run(self, request_factory, runs: int = 20) -> PerfSummary:
        latencies = []
        for _ in range(runs):
            request: GenerationRequest = request_factory()
            start = time.perf_counter()
            self.orchestrator.process(request)
            latencies.append((time.perf_counter() - start) * 1000)

        latencies.sort()

        def percentile(p: float) -> float:
            idx = int((len(latencies) - 1) * p)
            return latencies[idx]

        return PerfSummary(runs=runs, p95_latency_ms=percentile(0.95), p99_latency_ms=percentile(0.99))
