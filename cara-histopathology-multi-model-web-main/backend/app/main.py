"""FastAPI application for CARA multi-model histopathology analysis.

Run:  uvicorn app.main:app --reload      (from the backend/ folder)
Docs: http://localhost:8000/docs
"""
from __future__ import annotations

import io
import os
import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError

from .inference import InferenceEngine, ensure_samples, list_samples
from .utils import CLASS_NAMES, MODEL_KEYS, SAMPLES_DIR, class_display, ensure_dirs

MAX_UPLOAD_BYTES = int(os.environ.get("CARA_MAX_UPLOAD_MB", "15")) * 1024 * 1024
Image.MAX_IMAGE_PIXELS = 120_000_000   # guard against decompression bombs

engine: InferenceEngine | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine
    ensure_dirs()
    ensure_samples()
    engine = InferenceEngine()
    if os.environ.get("CARA_PRELOAD", "1") == "1":
        threading.Thread(target=engine.warmup, daemon=True).start()
    yield


app = FastAPI(title="CARA Histopathology Multi-Model API", version="1.0.0", lifespan=lifespan)

origins = [o.strip() for o in os.environ.get("ALLOWED_ORIGINS", "*").split(",") if o.strip()]
app.add_middleware(CORSMiddleware, allow_origins=origins, allow_methods=["*"], allow_headers=["*"])

SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/samples", StaticFiles(directory=str(SAMPLES_DIR)), name="samples")


def _engine() -> InferenceEngine:
    if engine is None:
        raise HTTPException(503, "Engine is still starting up")
    return engine


async def _read_image(file: UploadFile) -> Image.Image:
    data = await file.read()
    if not data:
        raise HTTPException(400, "Empty file")
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"File larger than {MAX_UPLOAD_BYTES // (1024 * 1024)} MB")
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError):
        raise HTTPException(400, "Could not decode the file as an image (use PNG, JPG, TIFF or BMP)")
    return img


@app.get("/api/health")
def health():
    return _engine().health()


@app.get("/api/classes")
def classes():
    return [{"index": i, "name": n, "label": class_display(n)} for i, n in enumerate(CLASS_NAMES)]


@app.get("/api/models")
def models():
    return _engine().model_list()


@app.get("/api/results")
def results():
    """Pre-computed test-set results of all models (global performance dashboard)."""
    return _engine().results


@app.get("/api/samples")
def samples():
    return [{**s, "url": f"/samples/{s['file']}"} for s in list_samples()]


@app.post("/api/compare")
async def compare(file: UploadFile = File(...), reference_class: str | None = Form(None)):
    """Run every model on the uploaded image and return side-by-side CARA results."""
    img = await _read_image(file)
    return _engine().analyze(img, MODEL_KEYS, reference_class)


@app.post("/api/predict/{model_key}")
async def predict(model_key: str, file: UploadFile = File(...), reference_class: str | None = Form(None)):
    if model_key not in MODEL_KEYS:
        raise HTTPException(404, f"Unknown model '{model_key}'. Choose one of {MODEL_KEYS}")
    img = await _read_image(file)
    return _engine().analyze(img, [model_key], reference_class)


@app.post("/api/explain/{model_key}")
async def explain(model_key: str, file: UploadFile = File(...), target_class: int | None = Form(None)):
    """Generate SHAP explanation for the specified model on the uploaded image."""
    if model_key not in MODEL_KEYS:
        raise HTTPException(404, f"Unknown model '{model_key}'. Choose one of {MODEL_KEYS}")
    img = await _read_image(file)
    return _engine().explain(img, model_key, target_class)


@app.get("/")
def root():
    return {"name": "CARA Histopathology Multi-Model API", "docs": "/docs", "health": "/api/health"}


@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    raise HTTPException(404)
