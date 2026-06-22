# YOLO COCO Test Benchmark

Web-based MVP for uploading a COCO-format test set, running YOLO inference, calculating overall/per-label mAP, and reviewing annotated case results.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run with a model path:

```bash
MODEL_PATH=/path/to/best.pt flask --app app run --debug
```

Or upload a `.pt` model directly in the web form.

## Input

- COCO JSON annotation file.
- Test images referenced by `images[].file_name` in the JSON.
- Optional YOLO `.pt` model if `MODEL_PATH` is not configured.
