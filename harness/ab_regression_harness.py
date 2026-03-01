from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RegressionSnapshot:
    identity_similarity_avg: float
    clip_score_avg: float
    artifact_score_avg: float
    p95_latency_ms: float


@dataclass
class ABRegressionResult:
    regressions: list[str]
    improved: list[str]


class ABRegressionHarness:
    def compare(self, baseline: RegressionSnapshot, candidate: RegressionSnapshot) -> ABRegressionResult:
        regressions: list[str] = []
        improved: list[str] = []

        if candidate.identity_similarity_avg < baseline.identity_similarity_avg - 0.01:
            regressions.append("identity_similarity_avg dropped > 0.01")
        elif candidate.identity_similarity_avg > baseline.identity_similarity_avg + 0.01:
            improved.append("identity_similarity_avg improved")

        if candidate.clip_score_avg < baseline.clip_score_avg - 0.01:
            regressions.append("clip_score_avg dropped > 0.01")
        elif candidate.clip_score_avg > baseline.clip_score_avg + 0.01:
            improved.append("clip_score_avg improved")

        if candidate.artifact_score_avg > baseline.artifact_score_avg + 0.01:
            regressions.append("artifact_score_avg worsened > 0.01")
        elif candidate.artifact_score_avg < baseline.artifact_score_avg - 0.01:
            improved.append("artifact_score_avg improved")

        if candidate.p95_latency_ms > baseline.p95_latency_ms + 500:
            regressions.append("p95_latency_ms increased > 500ms")
        elif candidate.p95_latency_ms < baseline.p95_latency_ms - 500:
            improved.append("p95_latency_ms improved")

        return ABRegressionResult(regressions=regressions, improved=improved)
