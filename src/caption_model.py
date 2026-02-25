from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


class CaptionGenerator(Protocol):
    def generate_caption(self, image_path: Path) -> str:
        ...


@dataclass
class HuggingFaceCaptionModel:
    model_name: str
    max_new_tokens: int = 30
    device: str | None = None
    _processor: object | None = None
    _model: object | None = None

    def __post_init__(self) -> None:
        self.device = self.device or self._pick_device()

    @staticmethod
    def _pick_device() -> str:
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"

    def _lazy_load(self) -> None:
        if self._processor is not None and self._model is not None:
            return

        from transformers import AutoProcessor, Blip2ForConditionalGeneration

        self._processor = AutoProcessor.from_pretrained(self.model_name)
        self._model = Blip2ForConditionalGeneration.from_pretrained(self.model_name)
        if hasattr(self._model, "to"):
            self._model.to(self.device)

    def generate_caption(self, image_path: Path) -> str:
        self._lazy_load()

        from PIL import Image

        image = Image.open(image_path).convert("RGB")
        inputs = self._processor(images=image, return_tensors="pt")
        if hasattr(inputs, "to"):
            inputs = inputs.to(self.device)
        outputs = self._model.generate(**inputs, max_new_tokens=self.max_new_tokens)
        caption = self._processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()
        return caption
