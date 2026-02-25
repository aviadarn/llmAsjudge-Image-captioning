from __future__ import annotations

import base64
import urllib.request
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


_DETERMINISTIC_IMAGE_B64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/a3sAAAAASUVORK5CYII="

_GOOGLE_IMAGE_URL = (
    "https://www.google.com/images/branding/googlelogo/2x/googlelogo_color_272x92dp.png"
)


def _write_deterministic_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(base64.b64decode(_DETERMINISTIC_IMAGE_B64))


def _try_download_google_image(path: Path) -> bool:
    try:
        with urllib.request.urlopen(_GOOGLE_IMAGE_URL, timeout=5) as response:
            content = response.read()
        if content:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
            return True
    except Exception:
        return False
    return False


@pytest.fixture
def sample_image(tmp_path_factory: pytest.TempPathFactory) -> Path:
    fixture_path = Path(__file__).parent / "fixtures" / "sample_image.jpg"
    if fixture_path.exists():
        return fixture_path

    temp_dir = tmp_path_factory.mktemp("generated_fixture")
    generated_image = temp_dir / "sample_image.jpg"
    downloaded = _try_download_google_image(generated_image)
    if not downloaded:
        _write_deterministic_image(generated_image)
    return generated_image


@pytest.fixture
def temp_output_dir(tmp_path: Path) -> Path:
    return tmp_path / "output"


@pytest.fixture
def mock_caption_model():
    model = Mock()
    model.generate_caption.side_effect = lambda image_path: f"caption for {Path(image_path).stem}"
    return model


@pytest.fixture
def mock_evaluator():
    evaluator = Mock()
    evaluator.evaluate.return_value = SimpleNamespace(
        g_eval_score=0.9,
        coherence_score=0.8,
        passed=True,
    )
    return evaluator
