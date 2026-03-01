from harness.ab_regression_harness import ABRegressionHarness, RegressionSnapshot
from harness.kpi_harness import KPIValidationHarness, KPIMetrics
from harness.load_harness import LoadTestingHarness


def test_kpi_harness_passes_when_all_targets_met() -> None:
    harness = KPIValidationHarness()
    result = harness.evaluate(
        KPIMetrics(
            identity_similarity=0.9,
            clip_score=0.85,
            face_distortion_rate=0.02,
            p95_latency_sec=8.0,
            rejection_rate=0.05,
            memory_leak_detected=False,
            all_tests_passed=True,
        )
    )
    assert result.passed
    assert result.failed_checks == []


def test_ab_regression_harness_detects_identity_regression() -> None:
    harness = ABRegressionHarness()
    result = harness.compare(
        baseline=RegressionSnapshot(
            identity_similarity_avg=0.9,
            clip_score_avg=0.84,
            artifact_score_avg=0.08,
            p95_latency_ms=4200,
        ),
        candidate=RegressionSnapshot(
            identity_similarity_avg=0.86,
            clip_score_avg=0.85,
            artifact_score_avg=0.07,
            p95_latency_ms=3900,
        ),
    )
    assert "identity_similarity_avg dropped > 0.01" in result.regressions


def test_load_harness_collects_percentiles() -> None:
    harness = LoadTestingHarness()
    calls = {"count": 0}

    def request_fn() -> None:
        calls["count"] += 1

    result = harness.run(request_fn=request_fn, total_requests=10, concurrency=3)
    assert result.total_requests == 10
    assert result.error_count == 0
    assert result.p95_latency_ms >= 0.0
    assert calls["count"] == 10
