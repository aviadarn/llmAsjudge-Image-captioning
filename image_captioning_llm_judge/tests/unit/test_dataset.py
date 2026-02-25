from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.dataset import load_coco_dataset, load_images_from_folder


def test_load_images_from_folder_non_recursive(tmp_path: Path, sample_image: Path) -> None:
    image_target = tmp_path / "image.jpg"
    image_target.write_bytes(sample_image.read_bytes())
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "nested.jpg").write_bytes(sample_image.read_bytes())

    samples = load_images_from_folder(tmp_path, recursive=False)

    assert len(samples) == 1
    assert samples[0].image_path == image_target


def test_load_images_from_folder_recursive(tmp_path: Path, sample_image: Path) -> None:
    nested = tmp_path / "nested"
    nested.mkdir()
    image_target = nested / "nested.jpg"
    image_target.write_bytes(sample_image.read_bytes())

    samples = load_images_from_folder(tmp_path, recursive=True)

    assert len(samples) == 1
    assert samples[0].image_path == image_target


def test_load_images_from_folder_errors(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_images_from_folder(tmp_path / "missing")

    text_file = tmp_path / "file.txt"
    text_file.write_text("not a folder", encoding="utf-8")
    with pytest.raises(NotADirectoryError):
        load_images_from_folder(text_file)


def test_load_coco_dataset_valid(tmp_path: Path, sample_image: Path) -> None:
    image_file = tmp_path / "img.jpg"
    image_file.write_bytes(sample_image.read_bytes())
    dataset = {
        "images": [{"id": 4, "file_name": "img.jpg"}],
        "annotations": [{"image_id": 4, "caption": "demo"}],
    }
    coco_path = tmp_path / "dataset.json"
    coco_path.write_text(json.dumps(dataset), encoding="utf-8")

    samples = load_coco_dataset(coco_path)

    assert len(samples) == 1
    assert samples[0].image_id == "4"
    assert samples[0].reference_caption == "demo"


def test_load_coco_dataset_errors(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_coco_dataset(tmp_path / "missing.json")

    malformed = tmp_path / "bad.json"
    malformed.write_text(json.dumps([1, 2, 3]), encoding="utf-8")
    with pytest.raises(ValueError, match="must be an object"):
        load_coco_dataset(malformed)

    missing_images = tmp_path / "missing_images.json"
    missing_images.write_text(json.dumps({"annotations": []}), encoding="utf-8")
    with pytest.raises(ValueError, match="missing 'images'"):
        load_coco_dataset(missing_images)

    wrong_images = tmp_path / "wrong_images.json"
    wrong_images.write_text(json.dumps({"images": {}}), encoding="utf-8")
    with pytest.raises(ValueError, match="must be a list"):
        load_coco_dataset(wrong_images)

    bad_item = tmp_path / "bad_item.json"
    bad_item.write_text(json.dumps({"images": [{"id": "x", "file_name": 1}]}), encoding="utf-8")
    with pytest.raises(ValueError, match="Each image item"):
        load_coco_dataset(bad_item)
