# image_captioning_llm_judge

Production-grade, deterministic image captioning + LLM-judge evaluation pipeline with both batch and API execution modes.

## Features

- HuggingFace image captioning (`HuggingFaceCaptionModel`) with configurable model name.
- DeepEval-based evaluation (`GEval` + `ImageCoherenceMetric`) with pass/fail thresholds.
- Batch processing pipeline that writes JSON and CSV artifacts.
- FastAPI API endpoint for online caption + evaluation.
- Full pytest suite (unit + integration), with deterministic mock-based tests.
- Docker and docker-compose for runtime and test execution.

## Project layout

```text
image_captioning_llm_judge/
├── src/
├── tests/
├── docker/
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── requirements-dev.txt
├── pytest.ini
├── Makefile
└── README.md
```

## Installation

```bash
make install
```

## Run tests

```bash
make test
```

Coverage is enforced at >= 85% via `pytest.ini`.

## Run API

```bash
make run
```

Then call:

```bash
curl -X POST "http://localhost:8000/caption" \
  -F "file=@tests/fixtures/sample_image.jpg"
```

## Batch pipeline usage

```python
from pathlib import Path

from src.caption_model import HuggingFaceCaptionModel
from src.evaluator import CaptionEvaluator
from src.pipeline import CaptioningPipeline

pipeline = CaptioningPipeline(
    caption_model=HuggingFaceCaptionModel(model_name="Salesforce/blip2-opt-2.7b"),
    evaluator=CaptionEvaluator(),
)

pipeline.run(
    input_path=Path("tests/fixtures"),
    output_dir=Path("outputs"),
    dataset_format="folder",
)
```

## TDD workflow

1. Add/modify tests in `tests/unit` or `tests/integration`.
2. Run `make test` and observe failure.
3. Implement changes in `src/`.
4. Re-run `make test` until green.
5. Run `make coverage` before merge.

### Writing new tests

- Keep business logic pure where possible.
- Inject dependencies instead of hard-coding them.
- Use `pytest-mock` to isolate side effects and heavy components.
- Use fixtures from `tests/conftest.py` for common test setup.

### How to mock DeepEval

You can inject fake metrics directly into `CaptionEvaluator`:

```python
class FakeMetric:
    def __init__(self, score):
        self.score = score
    def measure(self, **kwargs):
        return type("Score", (), {"score": self.score})()

evaluator = CaptionEvaluator(
    g_eval_metric=FakeMetric(0.9),
    coherence_metric=FakeMetric(0.8),
)
```

This avoids network/API calls and keeps tests deterministic.

### Deterministic testing strategy

- No real model downloads in unit/integration tests.
- No real LLM judge API calls.
- All heavy model interactions are mocked.
- Outputs are validated from static fixtures and controlled fake values.

## Docker

Build and run app:

```bash
make docker-build
docker compose up app
```

Run containerized tests:

```bash
make docker-test
```

## CI-ready command

Example CI command:

```bash
pip install -r requirements-dev.txt && pytest
```
