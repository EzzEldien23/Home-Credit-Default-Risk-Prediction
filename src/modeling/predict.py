from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from catboost import Pool
from scipy.stats import rankdata

from src.config import MODELS_DIR


DEFAULT_MODEL_PATH = MODELS_DIR / "home_credit_model.joblib"


def rank_norm(a):
    return rankdata(a) / len(a)


def load_model(path: str | Path = DEFAULT_MODEL_PATH) -> dict[str, Any]:
    return joblib.load(path)


def _avg_predict(models, frame):
    pred = np.zeros(len(frame))
    for model in models:
        pred += model.predict_proba(frame)[:, 1] / len(models)
    return pred


def predict_model_ready(frame: pd.DataFrame, artifact: dict[str, Any]) -> np.ndarray:
    frame = frame.copy().replace([np.inf, -np.inf], np.nan)
    features = artifact["features"]
    frame = frame.reindex(columns=features, fill_value=np.nan)
    base_predictions = {}
    for name in ["lgb_a", "lgb_b", "lgb_seed", "xgb"]:
        spec = artifact["models"][name]
        base_predictions[name] = _avg_predict(spec["models"], frame[spec["features"]])

    cat_spec = artifact["models"]["cat"]
    cat_frame = frame.reindex(columns=cat_spec["features"], fill_value=np.nan)
    for col in cat_spec["cat_features"]:
        if col in cat_frame.columns:
            cat_frame[col] = cat_frame[col].fillna("__nan__").astype(str)
    cat_pool = Pool(cat_frame, cat_features=[cat_frame.columns.get_loc(c) for c in cat_spec["cat_features"] if c in cat_frame.columns])
    cat_pred = np.zeros(len(cat_frame))
    for model in cat_spec["models"]:
        cat_pred += model.predict_proba(cat_pool)[:, 1] / len(cat_spec["models"])
    base_predictions["cat"] = cat_pred

    stack_spec = artifact["models"]["stack_lr"]
    stack_input = np.column_stack([base_predictions[n] for n in stack_spec["base_names"]])
    stack_pred = np.zeros(len(frame))
    for model in stack_spec["models"]:
        stack_pred += model.predict_proba(stack_input)[:, 1] / len(stack_spec["models"])
    base_predictions["stack_lr"] = stack_pred

    ranked = [rank_norm(base_predictions[name]) for name in artifact["blend_names"]]
    weights = artifact["blend_weights"]
    return sum(w * p for w, p in zip(weights, ranked))


def predict_csv(input_csv: str | Path, output_csv: str | Path, model_path: str | Path = DEFAULT_MODEL_PATH):
    artifact = load_model(model_path)
    frame = pd.read_csv(input_csv)
    ids = frame["SK_ID_CURR"] if "SK_ID_CURR" in frame.columns else pd.Series(range(len(frame)), name="row_id")
    preds = predict_model_ready(frame, artifact)
    out = pd.DataFrame({ids.name: ids, "TARGET": preds})
    out.to_csv(output_csv, index=False)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="models/predictions.csv")
    parser.add_argument("--model", default=str(DEFAULT_MODEL_PATH))
    args = parser.parse_args()
    predict_csv(args.input, args.output, args.model)


if __name__ == "__main__":
    main()
