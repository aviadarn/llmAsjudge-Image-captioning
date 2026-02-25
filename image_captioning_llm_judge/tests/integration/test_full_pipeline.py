from __future__ import annotations

import csv
from pathlib import Path
from types import SimpleNamespace

from src.pipeline import CaptioningPipeline


class MockCaptionModel:
    def generate_caption(self, image_path: Path) -> str:
        return f"caption::{image_path.stem}"


class MockEvaluator:
    def evaluate(self, image_caption: str, reference_caption: str | None = None) -> SimpleNamespace:  # noqa: ARG002
        return SimpleNamespace(g_eval_score=0.95, coherence_score=0.9, passed=True)


def test_full_pipeline_end_to_end(tmp_path: Path, sample_image: Path) -> None:
    image = tmp_path / "img.jpg"
    image.write_bytes(sample_image.read_bytes())

    pipeline = CaptioningPipeline(caption_model=MockCaptionModel(), evaluator=MockEvaluator())
    output_dir = tmp_path / "output"

    records = pipeline.run(input_path=tmp_path, output_dir=output_dir, dataset_format="folder")

    assert len(records) == 1
    assert records[0].caption == "caption::img"

    with (output_dir / "results.csv").open(encoding="utf-8") as csv_file:
        rows = list(csv.DictReader(csv_file))
    assert rows[0]["passed"] in ["True", "1", "true"]
