from __future__ import annotations

from dataclasses import dataclass


@dataclass
class KPIMetrics:
    identity_similarity: float
    clip_score: float
    face_distortion_rate: float
    p95_latency_sec: float
    rejection_rate: float
    memory_leak_detected: bool
    all_tests_passed: bool


@dataclass
class KPIEvaluation:
    passed: bool
    failed_checks: list[str]


class KPIValidationHarness:
    def evaluate(self, metrics: KPIMetrics) -> KPIEvaluation:
        failed: list[str] = []

        if metrics.identity_similarity < 0.85:
            failed.append("identity_similarity < 0.85")
        if metrics.clip_score < 0.8:
            failed.append("clip_score < 0.8")
        if metrics.face_distortion_rate >= 0.03:
            failed.append("face_distortion_rate >= 3%")
        if metrics.p95_latency_sec >= 10.0:
            failed.append("p95_latency >= 10 sec")
        if metrics.rejection_rate >= 0.08:
            failed.append("rejection_rate >= 8%")
        if metrics.memory_leak_detected:
            failed.append("memory_leak_detected = true")
        if not metrics.all_tests_passed:
            failed.append("harness tests are not fully passed")

        return KPIEvaluation(passed=len(failed) == 0, failed_checks=failed)
