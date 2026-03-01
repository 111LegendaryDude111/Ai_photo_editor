from __future__ import annotations

import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Callable


@dataclass
class LoadTestResult:
    total_requests: int
    concurrency: int
    p95_latency_ms: float
    p99_latency_ms: float
    error_count: int


class LoadTestingHarness:
    def run(self, request_fn: Callable[[], None], total_requests: int = 50, concurrency: int = 8) -> LoadTestResult:
        latencies: list[float] = []
        errors = 0

        def timed_call() -> float:
            start = time.perf_counter()
            request_fn()
            return (time.perf_counter() - start) * 1000.0

        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = [pool.submit(timed_call) for _ in range(total_requests)]
            for future in as_completed(futures):
                try:
                    latencies.append(float(future.result()))
                except Exception:
                    errors += 1

        latencies.sort()
        if not latencies:
            return LoadTestResult(total_requests=total_requests, concurrency=concurrency, p95_latency_ms=0.0, p99_latency_ms=0.0, error_count=errors)

        p95_index = max(0, int((len(latencies) - 1) * 0.95))
        p99_index = max(0, int((len(latencies) - 1) * 0.99))
        p95 = latencies[p95_index]
        p99 = latencies[p99_index]

        # Keep a small side effect so harness users can quickly inspect average latency too.
        _ = statistics.mean(latencies)

        return LoadTestResult(
            total_requests=total_requests,
            concurrency=concurrency,
            p95_latency_ms=p95,
            p99_latency_ms=p99,
            error_count=errors,
        )
