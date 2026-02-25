from __future__ import annotations

from types import SimpleNamespace

from src.config import EvalThresholds
from src.evaluator import CaptionEvaluator


class FakeMetric:
    def __init__(self, score: float):
        self.score = score

    def measure(self, **kwargs):  # noqa: ARG002
        return SimpleNamespace(score=self.score)


def test_evaluator_structured_output() -> None:
    evaluator = CaptionEvaluator(
        g_eval_metric=FakeMetric(0.8),
        coherence_metric=FakeMetric(0.7),
        thresholds=EvalThresholds(g_eval_min_score=0.5, coherence_min_score=0.5),
    )

    result = evaluator.evaluate(image_caption="a cat on a chair", reference_caption="cat on chair")

    assert result.g_eval_score == 0.8
    assert result.coherence_score == 0.7
    assert result.passed is True


def test_evaluator_threshold_fail() -> None:
    evaluator = CaptionEvaluator(
        g_eval_metric=FakeMetric(0.3),
        coherence_metric=FakeMetric(0.9),
        thresholds=EvalThresholds(g_eval_min_score=0.5, coherence_min_score=0.5),
    )

    result = evaluator.evaluate(image_caption="bad caption")

    assert result.passed is False


def test_evaluator_extract_numeric_scores() -> None:
    class NumericMetric:
        def __init__(self, value: float):
            self.value = value

        def measure(self, **kwargs):  # noqa: ARG002
            return self.value

    evaluator = CaptionEvaluator(g_eval_metric=NumericMetric(1.0), coherence_metric=NumericMetric(0.6))
    result = evaluator.evaluate(image_caption="test")
    assert result.g_eval_score == 1.0
    assert result.coherence_score == 0.6
