from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .utils import is_image_file


@dataclass(frozen=True)
class ImageSample:
    image_id: str
    image_path: Path
    reference_caption: str | None = None


def load_images_from_folder(folder: Path, recursive: bool = False) -> list[ImageSample]:
    if not folder.exists():
        raise FileNotFoundError(f"Input folder does not exist: {folder}")
    if not folder.is_dir():
        raise NotADirectoryError(f"Input path is not a directory: {folder}")

    iterator = folder.rglob("*") if recursive else folder.glob("*")
    samples = [
        ImageSample(image_id=path.stem, image_path=path)
        for path in sorted(iterator)
        if path.is_file() and is_image_file(path)
    ]
    return samples


def _validate_coco_json(payload: dict[str, Any]) -> None:
    if "images" not in payload:
        raise ValueError("COCO JSON missing 'images' field")
    if not isinstance(payload["images"], list):
        raise ValueError("COCO JSON field 'images' must be a list")


def load_coco_dataset(coco_path: Path, image_root: Path | None = None) -> list[ImageSample]:
    if not coco_path.exists():
        raise FileNotFoundError(f"COCO file does not exist: {coco_path}")

    payload = json.loads(coco_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("COCO JSON must be an object")
    _validate_coco_json(payload)

    caption_map: dict[int, str] = {}
    for annotation in payload.get("annotations", []):
        image_id = annotation.get("image_id")
        caption = annotation.get("caption")
        if isinstance(image_id, int) and isinstance(caption, str) and image_id not in caption_map:
            caption_map[image_id] = caption

    base = image_root if image_root is not None else coco_path.parent
    samples: list[ImageSample] = []
    for image in payload["images"]:
        image_id = image.get("id")
        file_name = image.get("file_name")
        if not isinstance(image_id, int) or not isinstance(file_name, str):
            raise ValueError("Each image item must have integer 'id' and string 'file_name'")
        image_path = (base / file_name).resolve()
        samples.append(
            ImageSample(
                image_id=str(image_id),
                image_path=image_path,
                reference_caption=caption_map.get(image_id),
            )
        )

    return samples
