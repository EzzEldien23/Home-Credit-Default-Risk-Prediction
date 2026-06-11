from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"


TABLE_FILES = {
    "application_train": "application_train.csv",
    "application_test": "application_test.csv",
    "bureau": "bureau.csv",
    "bureau_balance": "bureau_balance.csv",
    "previous_application": "previous_application.csv",
    "pos_cash_balance": "POS_CASH_balance.csv",
    "credit_card_balance": "credit_card_balance.csv",
    "installments_payments": "installments_payments.csv",
    "sample_submission": "sample_submission.csv",
}


@dataclass
class TrainingConfig:
    seed: int = 42
    n_folds: int = 5
    te_smoothing: int = 40
    te_min_samples: int = 80
    null_importance_runs: int = 5
    null_importance_sample: int | None = 60000
    correlation_sample: int | None = 30000
    top_feature_count: int = 400
    knn_neighbors: list[int] = field(default_factory=lambda: [200, 500])
    lgb_seed_average_seeds: list[int] = field(default_factory=lambda: [456, 789, 1234])
    row_limits: dict[str, int | None] = field(default_factory=dict)
    output_submission: Path = MODELS_DIR / "submission.csv"
    model_artifact: Path = MODELS_DIR / "home_credit_model.joblib"
    feature_metadata: Path = MODELS_DIR / "feature_metadata.json"
    metrics_path: Path = REPORTS_DIR / "metrics.json"
    experiment_name: str = "home-credit-default-risk"
    mlflow_tracking_uri: str | None = None
    dagshub_repo_owner: str | None = None
    dagshub_repo_name: str | None = None

    def row_limit(self, table_name: str) -> int | None:
        value = self.row_limits.get(table_name)
        if value in ("", "null", "None"):
            return None
        return int(value) if value is not None else None


def _coerce_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_params(path: str | Path = PROJECT_ROOT / "params.yaml") -> dict[str, Any]:
    path = _coerce_path(path)
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_training_config(path: str | Path = PROJECT_ROOT / "params.yaml") -> TrainingConfig:
    params = load_params(path)
    train_params = params.get("train", {})
    mlflow_params = params.get("mlflow", {})
    output_params = params.get("outputs", {})
    cfg = TrainingConfig(
        seed=train_params.get("seed", 42),
        n_folds=train_params.get("n_folds", 5),
        te_smoothing=train_params.get("te_smoothing", 40),
        te_min_samples=train_params.get("te_min_samples", 80),
        null_importance_runs=train_params.get("null_importance_runs", 5),
        null_importance_sample=train_params.get("null_importance_sample", 60000),
        correlation_sample=train_params.get("correlation_sample", 30000),
        top_feature_count=train_params.get("top_feature_count", 400),
        knn_neighbors=train_params.get("knn_neighbors", [200, 500]),
        lgb_seed_average_seeds=train_params.get("lgb_seed_average_seeds", [456, 789, 1234]),
        row_limits=params.get("row_limits", {}),
        output_submission=_coerce_path(output_params.get("submission", MODELS_DIR / "submission.csv")),
        model_artifact=_coerce_path(
            output_params.get("model_artifact", MODELS_DIR / "home_credit_model.joblib")
        ),
        feature_metadata=_coerce_path(
            output_params.get("feature_metadata", MODELS_DIR / "feature_metadata.json")
        ),
        metrics_path=_coerce_path(output_params.get("metrics", REPORTS_DIR / "metrics.json")),
        experiment_name=mlflow_params.get("experiment_name", "home-credit-default-risk"),
        mlflow_tracking_uri=mlflow_params.get("tracking_uri"),
        dagshub_repo_owner=mlflow_params.get("dagshub_repo_owner"),
        dagshub_repo_name=mlflow_params.get("dagshub_repo_name"),
    )
    return cfg
