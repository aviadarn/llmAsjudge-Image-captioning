from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from .config import EvalThresholds


class JudgeMetric(Protocol):
    def measure(self, **kwargs: Any) -> Any:
        ...


@dataclass(frozen=True)
class EvaluationResult:
    g_eval_score: float
    coherence_score: float
    passed: bool


class CaptionEvaluator:
    def __init__(
        self,
        g_eval_metric: JudgeMetric | None = None,
        coherence_metric: JudgeMetric | None = None,
        thresholds: EvalThresholds | None = None,
    ) -> None:
        self.thresholds = thresholds or EvalThresholds()
        self.g_eval_metric = g_eval_metric or self._default_g_eval()
        self.coherence_metric = coherence_metric or self._default_coherence()

    @staticmethod
    def _default_g_eval() -> JudgeMetric:
        from deepeval.metrics import GEval

        return GEval(
            name="CaptionQuality",
            criteria="Score caption quality from 0-1 based on relevance and fluency",
            evaluation_params=[],
        )

    @staticmethod
    def _default_coherence() -> JudgeMetric:
        from deepeval.metrics.multimodal_metrics.image_coherence.image_coherence import (
            ImageCoherenceMetric,
        )

        return ImageCoherenceMetric()

    @staticmethod
    def _extract_score(metric_response: Any) -> float:
        if isinstance(metric_response, (int, float)):
            return float(metric_response)
        score = getattr(metric_response, "score", None)
        if isinstance(score, (int, float)):
            return float(score)
        raise ValueError("Metric response does not expose a numeric score")

    def evaluate(self, image_caption: str, reference_caption: str | None = None) -> EvaluationResult:
        g_eval_input = {"actual_output": image_caption}
        if reference_caption:
            g_eval_input["expected_output"] = reference_caption

        g_eval_raw = self.g_eval_metric.measure(**g_eval_input)
        coherence_raw = self.coherence_metric.measure(caption=image_caption)

        g_eval_score = self._extract_score(g_eval_raw)
        coherence_score = self._extract_score(coherence_raw)
        passed = (
            g_eval_score >= self.thresholds.g_eval_min_score
            and coherence_score >= self.thresholds.coherence_min_score
        )
        return EvaluationResult(
            g_eval_score=g_eval_score,
            coherence_score=coherence_score,
            passed=passed,
        )
