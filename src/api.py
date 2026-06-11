from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from src.config import MODELS_DIR
from src.modeling.predict import load_model, predict_model_ready


MODEL_PATH = Path(os.getenv("MODEL_PATH", MODELS_DIR / "home_credit_model.joblib"))

app = FastAPI(title="Home Credit Default Risk API", version="0.1.0")
_model = None


class PredictionRows(BaseModel):
    rows: list[dict] = Field(..., description="Model-ready feature rows.")


def get_model():
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise HTTPException(status_code=503, detail=f"Model artifact not found at {MODEL_PATH}")
        _model = load_model(MODEL_PATH)
    return _model


@app.get("/health")
def health():
    return {"status": "ok", "model_path": str(MODEL_PATH), "model_exists": MODEL_PATH.exists()}


@app.post("/predict")
def predict(payload: PredictionRows):
    if not payload.rows:
        raise HTTPException(status_code=400, detail="No rows supplied.")
    frame = pd.DataFrame(payload.rows)
    preds = predict_model_ready(frame, get_model())
    return {"predictions": preds.tolist()}


@app.post("/predict-file")
async def predict_file(file: UploadFile = File(...)):
    frame = pd.read_csv(file.file)
    preds = predict_model_ready(frame, get_model())
    ids = frame["SK_ID_CURR"].tolist() if "SK_ID_CURR" in frame.columns else list(range(len(frame)))
    return {"predictions": [{"id": row_id, "target": float(pred)} for row_id, pred in zip(ids, preds)]}
