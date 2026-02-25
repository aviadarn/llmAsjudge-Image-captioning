from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

from .dataset import ImageSample, load_coco_dataset, load_images_from_folder
from .evaluator import CaptionEvaluator, EvaluationResult
from .utils import ensure_dir, write_json


class CaptionGenerator(Protocol):
    def generate_caption(self, image_path: Path) -> str:
        ...


@dataclass(frozen=True)
class PipelineRecord:
    image_id: str
    image_path: str
    caption: str
    reference_caption: str | None
    g_eval_score: float
    coherence_score: float
    passed: bool


class CaptioningPipeline:
    def __init__(
        self,
        caption_model: CaptionGenerator,
        evaluator: CaptionEvaluator,
    ) -> None:
        self.caption_model = caption_model
        self.evaluator = evaluator

    def _load_dataset(self, input_path: Path, dataset_format: str, recursive: bool = False) -> list[ImageSample]:
        if dataset_format == "folder":
            return load_images_from_folder(input_path, recursive=recursive)
        if dataset_format == "coco":
            return load_coco_dataset(input_path)
        raise ValueError(f"Unsupported dataset_format: {dataset_format}")

    def run(
        self,
        input_path: Path,
        output_dir: Path,
        dataset_format: str = "folder",
        recursive: bool = False,
        fail_fast: bool = False,
    ) -> list[PipelineRecord]:
        ensure_dir(output_dir)
        samples = self._load_dataset(input_path=input_path, dataset_format=dataset_format, recursive=recursive)
        records: list[PipelineRecord] = []

        for sample in samples:
            try:
                caption = self.caption_model.generate_caption(sample.image_path)
                eval_result: EvaluationResult = self.evaluator.evaluate(
                    image_caption=caption,
                    reference_caption=sample.reference_caption,
                )
                records.append(
                    PipelineRecord(
                        image_id=sample.image_id,
                        image_path=str(sample.image_path),
                        caption=caption,
                        reference_caption=sample.reference_caption,
                        g_eval_score=eval_result.g_eval_score,
                        coherence_score=eval_result.coherence_score,
                        passed=eval_result.passed,
                    )
                )
            except Exception:
                if fail_fast:
                    raise

        self._write_outputs(records, output_dir)
        return records

    @staticmethod
    def _write_outputs(records: list[PipelineRecord], output_dir: Path) -> None:
        rows = [asdict(record) for record in records]
        json_path = output_dir / "results.json"
        csv_path = output_dir / "results.csv"
        write_json(json_path, rows)

        headers = [
            "image_id",
            "image_path",
            "caption",
            "reference_caption",
            "g_eval_score",
            "coherence_score",
            "passed",
        ]
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            writer.writerows(rows)
