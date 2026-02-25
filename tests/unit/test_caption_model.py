from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

from src.caption_model import HuggingFaceCaptionModel


class FakeTensorDict(dict):
    def to(self, _device: str):
        return self


class FakeProcessor:
    def __call__(self, images, return_tensors):  # noqa: ARG002
        return FakeTensorDict({"pixel_values": [1]})

    def batch_decode(self, outputs, skip_special_tokens):  # noqa: ARG002
        return ["a mocked caption"]


class FakeModel:
    def __init__(self):
        self.device = None

    def to(self, device: str):
        self.device = device

    def generate(self, **kwargs):  # noqa: ARG002
        return [[101, 102]]


def test_generate_caption_returns_string(tmp_path: Path, sample_image: Path, monkeypatch) -> None:
    image_target = tmp_path / "img.jpg"
    image_target.write_bytes(sample_image.read_bytes())

    fake_pil = SimpleNamespace(
        Image=SimpleNamespace(open=lambda path: SimpleNamespace(convert=lambda mode: f"image::{path}::{mode}"))
    )
    monkeypatch.setitem(sys.modules, "PIL", fake_pil)

    fake_model = FakeModel()
    caption_model = HuggingFaceCaptionModel(model_name="fake", _processor=FakeProcessor(), _model=fake_model)
    caption = caption_model.generate_caption(image_target)

    assert isinstance(caption, str)
    assert caption == "a mocked caption"


def test_pick_device_cuda_available(monkeypatch) -> None:
    fake_torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True))
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    assert HuggingFaceCaptionModel._pick_device() == "cuda"


def test_pick_device_fallback_cpu(monkeypatch) -> None:
    class Boom:
        class cuda:
            @staticmethod
            def is_available() -> bool:
                raise RuntimeError("boom")

    monkeypatch.setitem(sys.modules, "torch", Boom)
    assert HuggingFaceCaptionModel._pick_device() == "cpu"
