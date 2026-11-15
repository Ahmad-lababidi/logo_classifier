import sys
import time
import json
import joblib
import numpy as np
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Literal

APP_DIR = Path(__file__).resolve().parent
ROOT_DIR = APP_DIR if (APP_DIR / "saved_models").is_dir() else APP_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import torch
import torch.nn.functional as F
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Query, status

from src.models import DenseLogoCNN, resnet_model
from src.preprocessing import preprocess_image_bytes
from src.utils import (
    logger, load_checkpoint, load_class_names,
    verify_api_key, RateLimiter
)

MODELS = {}
CLASS_NAMES = []
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ALLOWED_MIME_TYPES = {"image/png", "image/jpeg", "image/jpg", "image/webp", "image/bmp"}
MAX_UPLOAD_BYTES = 10 * 1024 * 1024


@asynccontextmanager
async def lifespan(app: FastAPI):
    global CLASS_NAMES
    model_dir = ROOT_DIR / "saved_models"
    CLASS_NAMES = load_class_names(str(model_dir / "class_names.json"))
    num_classes = len(CLASS_NAMES)
    if num_classes < 2:
        raise RuntimeError("At least two classes are required; train the models first.")

    scratch = DenseLogoCNN(num_classes=num_classes)
    MODELS["scratch"] = load_checkpoint(scratch, str(model_dir / "scratch_cnn.pth"), DEVICE)
    MODELS["scratch_knn"] = joblib.load(model_dir / "scratch_knn.joblib")
    resnet = resnet_model(num_classes=num_classes, pretrained=False)
    MODELS["resnet"] = load_checkpoint(resnet, str(model_dir / "resnet_finetuned.pth"), DEVICE)
    logger.info("All three classifiers loaded on %s for %d classes.", DEVICE, num_classes)
    yield
    MODELS.clear()


app = FastAPI(
    title="Multi-Class Logo Classification API",
    version="1.0.0",
    lifespan=lifespan
)

rate_limiter = RateLimiter(max_requests=20, window_seconds=60)


def run_inference(model: torch.nn.Module, tensor: torch.Tensor, top_k: int):
    start = time.perf_counter()
    with torch.no_grad():
        logits = model(tensor.to(DEVICE))
        probs = F.softmax(logits, dim=1)[0]
        top_probs, top_indices = torch.topk(probs, k=min(top_k, len(CLASS_NAMES)))

    latency_ms = round((time.perf_counter() - start) * 1000, 2)
    predictions = [
        {"label": CLASS_NAMES[idx.item()], "confidence": round(prob.item(), 4)}
        for prob, idx in zip(top_probs, top_indices)
    ]
    return {"predictions": predictions, "inference_time_ms": latency_ms}


def run_knn_inference(cnn_model: DenseLogoCNN, knn_model, tensor: torch.Tensor, top_k: int):
    start = time.perf_counter()
    with torch.inference_mode():
        embedding = cnn_model.extract_features(tensor.to(DEVICE)).cpu().numpy()

    probabilities = knn_model.predict_proba(embedding)[0]
    indices = np.argsort(probabilities)[::-1][:min(top_k, len(probabilities))]
    predictions = [
        {
            "label": CLASS_NAMES[int(knn_model.classes_[index])],
            "confidence": round(float(probabilities[index]), 4),
        }
        for index in indices
    ]
    latency_ms = round((time.perf_counter() - start) * 1000, 2)
    return {"predictions": predictions, "inference_time_ms": latency_ms}


@app.get("/health")
async def health_check():
    return {
        "status": "healthy" if {"scratch", "scratch_knn", "resnet"}.issubset(MODELS) else "not_ready",
        "device": str(DEVICE),
        "models_loaded": list(MODELS.keys()),
        "num_classes": len(CLASS_NAMES)
    }


@app.get("/metrics", dependencies=[Depends(verify_api_key)])
async def model_metrics():
    metrics_path = ROOT_DIR / "saved_models" / "training_metrics.json"
    try:
        with metrics_path.open("r", encoding="utf-8") as metrics_file:
            metrics = json.load(metrics_file)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Training metrics are not available.")
    except (OSError, json.JSONDecodeError):
        logger.exception("Could not read saved training metrics.")
        raise HTTPException(status_code=500, detail="Could not load training metrics.")

    return {
        "seed": metrics.get("seed"),
        "validation_split": metrics.get("validation_split"),
        "class_count": metrics.get("class_count"),
        "training_images": metrics.get("training_images"),
        "validation_images": metrics.get("validation_images"),
        "unique_images": metrics.get("unique_images"),
        "models": {
            name: {
                "accuracy": model_metrics.get("accuracy"),
                "macro_f1": model_metrics.get("macro_f1"),
            }
            for name, model_metrics in (
                ("scratch_cnn", metrics.get("scratch_cnn", {})),
                ("scratch_knn", metrics.get("scratch_knn", {})),
                ("resnet", metrics.get("resnet", {})),
            )
        },
    }


@app.post(
    "/predict",
    dependencies=[Depends(verify_api_key), Depends(rate_limiter)]
)
async def predict_logo(
    file: UploadFile = File(...),
    model_type: Literal["scratch", "scratch_knn", "resnet", "all"] = Query("all"),
    top_k: int = Query(3, ge=1, le=10)
):
    if file.content_type not in ALLOWED_MIME_TYPES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid file type '{file.content_type}'. Allowed: {ALLOWED_MIME_TYPES}"
        )

    if not MODELS:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Models are not loaded. Run src/train.py first."
        )

    try:
        raw_bytes = await file.read(MAX_UPLOAD_BYTES + 1)
        if len(raw_bytes) > MAX_UPLOAD_BYTES:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail="Image exceeds the 10 MB upload limit.",
            )
        if len(raw_bytes) == 0:
            raise ValueError("Uploaded file is empty.")
        input_tensor = preprocess_image_bytes(raw_bytes)
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected preprocessing error: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Error processing image.")

    results = {}
    if model_type in ("scratch", "all"):
        results["scratch_cnn"] = run_inference(MODELS["scratch"], input_tensor, top_k)

    if model_type in ("scratch_knn", "all"):
        results["scratch_knn"] = run_knn_inference(
            MODELS["scratch"], MODELS["scratch_knn"], input_tensor, top_k
        )

    if model_type in ("resnet", "all"):
        results["resnet_finetuned"] = run_inference(MODELS["resnet"], input_tensor, top_k)

    return {
        "filename": file.filename,
        "results": results
    }
