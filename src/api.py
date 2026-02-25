from __future__ import annotations

import tempfile
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

from .caption_model import HuggingFaceCaptionModel
from .evaluator import CaptionEvaluator

app = FastAPI(title="Image Captioning LLM Judge")
caption_model = HuggingFaceCaptionModel(model_name="Salesforce/blip2-opt-2.7b")
evaluator = CaptionEvaluator()


class CaptionResponse(BaseModel):
    caption: str
    g_eval_score: float
    coherence_score: float
    passed: bool


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/caption", response_model=CaptionResponse)
async def caption_endpoint(file: UploadFile = File(...)) -> CaptionResponse:
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing filename")

    suffix = Path(file.filename).suffix or ".jpg"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp:
        content = await file.read()
        temp.write(content)
        temp_path = Path(temp.name)

    try:
        caption = caption_model.generate_caption(temp_path)
        result = evaluator.evaluate(image_caption=caption)
        return CaptionResponse(
            caption=caption,
            g_eval_score=result.g_eval_score,
            coherence_score=result.coherence_score,
            passed=result.passed,
        )
    finally:
        if temp_path.exists():
            temp_path.unlink()
