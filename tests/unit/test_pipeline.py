from __future__ import annotations

import csv
import json
from pathlib import Path

from src.pipeline import CaptioningPipeline


def test_pipeline_full_loop_and_outputs(tmp_path: Path, sample_image: Path, mock_caption_model, mock_evaluator) -> None:
    image1 = tmp_path / "a.jpg"
    image2 = tmp_path / "b.jpg"
    image1.write_bytes(sample_image.read_bytes())
    image2.write_bytes(sample_image.read_bytes())

    pipeline = CaptioningPipeline(caption_model=mock_caption_model, evaluator=mock_evaluator)
    output_dir = tmp_path / "out"

    records = pipeline.run(input_path=tmp_path, output_dir=output_dir, dataset_format="folder")

    assert len(records) == 2
    assert (output_dir / "results.json").exists()
    assert (output_dir / "results.csv").exists()

    data = json.loads((output_dir / "results.json").read_text(encoding="utf-8"))
    assert len(data) == 2

    with (output_dir / "results.csv").open(encoding="utf-8") as csv_file:
        rows = list(csv.DictReader(csv_file))
    assert len(rows) == 2


def test_pipeline_skips_errors(tmp_path: Path, sample_image: Path, mock_evaluator) -> None:
    ok = tmp_path / "ok.jpg"
    bad = tmp_path / "bad.jpg"
    ok.write_bytes(sample_image.read_bytes())
    bad.write_bytes(sample_image.read_bytes())

    class Model:
        def generate_caption(self, image_path: Path) -> str:
            if image_path.stem == "bad":
                raise RuntimeError("boom")
            return "ok caption"

    pipeline = CaptioningPipeline(caption_model=Model(), evaluator=mock_evaluator)
    records = pipeline.run(input_path=tmp_path, output_dir=tmp_path / "out", dataset_format="folder")

    assert len(records) == 1
    assert records[0].image_id == "ok"
