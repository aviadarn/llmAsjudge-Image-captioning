from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class CaptionModelConfig:
    """Runtime configuration for the captioning model."""

    model_name: str = "Salesforce/blip2-opt-2.7b"
    max_new_tokens: int = 30
    device: Optional[str] = None


@dataclass(frozen=True)
class PipelineConfig:
    """Batch pipeline configuration."""

    input_path: Path
    output_dir: Path
    dataset_format: str = "folder"
    recursive: bool = False
    fail_fast: bool = False
    caption_model: CaptionModelConfig = field(default_factory=CaptionModelConfig)


@dataclass(frozen=True)
class EvalThresholds:
    """Pass/fail thresholds for evaluator output."""

    g_eval_min_score: float = 0.5
    coherence_min_score: float = 0.5
