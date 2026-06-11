from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import asdict
from pathlib import Path
from typing import Any

import joblib
import lightgbm as lgb
import mlflow
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

from src.config import RAW_DATA_DIR, TrainingConfig, load_training_config
from src.features import build_feature_frame


def rank_norm(a):
    return rankdata(a) / len(a)


def prepare_model_matrices(train: pd.DataFrame, test: pd.DataFrame, config: TrainingConfig) -> dict[str, Any]:
    target = train["TARGET"].astype("int8")
    train_ids = train["SK_ID_CURR"].copy()
    test_ids = test["SK_ID_CURR"].copy()
    cat_train = train.drop(columns=["SK_ID_CURR"]).copy()
    cat_test = test.drop(columns=["SK_ID_CURR"]).copy()
    cat_feature_names = [c for c in cat_train.columns if c != "TARGET" and cat_train[c].dtype == "object"]
    for col in cat_feature_names:
        cat_train[col] = cat_train[col].fillna("__nan__").astype(str)
        cat_test[col] = cat_test[col].fillna("__nan__").astype(str)

    train_num = train.drop(columns=["TARGET", "SK_ID_CURR"])
    test_num = test.drop(columns=["SK_ID_CURR"])
    encoders = {}
    for col in [c for c in train_num.columns if train_num[c].dtype == "object"]:
        le = LabelEncoder()
        all_vals = pd.concat([train_num[col], test_num[col]], axis=0).astype(str).fillna("nan")
        le.fit(all_vals)
        train_num[col] = le.transform(train_num[col].astype(str).fillna("nan")).astype("int32")
        test_num[col] = le.transform(test_num[col].astype(str).fillna("nan")).astype("int32")
        encoders[col] = le

    train_num = train_num.replace([np.inf, -np.inf], np.nan)
    test_num = test_num.replace([np.inf, -np.inf], np.nan)
    cat_train = cat_train.replace([np.inf, -np.inf], np.nan)
    cat_test = cat_test.replace([np.inf, -np.inf], np.nan)
    train_num, test_num = train_num.align(test_num, join="inner", axis=1)

    cat_train_features = cat_train.drop(columns=["TARGET"]).copy()
    cat_train_features, cat_test = cat_train_features.align(cat_test, join="inner", axis=1)
    cat_train = pd.concat(
        [cat_train[["TARGET"]].reset_index(drop=True), cat_train_features.reset_index(drop=True)],
        axis=1,
    )
    cat_feature_names = [c for c in cat_feature_names if c in cat_train.columns]
    for col in cat_train.columns:
        if col not in cat_feature_names and col != "TARGET":
            cat_train[col] = pd.to_numeric(cat_train[col], errors="coerce")
            cat_test[col] = pd.to_numeric(cat_test[col], errors="coerce")

    clean_names = {c: re.sub(r"[^A-Za-z0-9_]+", "_", c) for c in train_num.columns}
    train_num = train_num.rename(columns=clean_names)
    test_num = test_num.rename(columns=clean_names)
    cat_clean_names = {c: re.sub(r"[^A-Za-z0-9_]+", "_", c) for c in cat_train.columns}
    cat_train = cat_train.rename(columns=cat_clean_names)
    cat_test = cat_test.rename(columns=cat_clean_names)
    cat_feature_names = [cat_clean_names.get(c, c) for c in cat_feature_names]

    drop_cols = [c for c in train_num.columns if train_num[c].isna().all() or train_num[c].nunique(dropna=False) <= 1]
    if drop_cols:
        train_num = train_num.drop(columns=drop_cols)
        test_num = test_num.drop(columns=drop_cols)
        cat_drop = [c for c in drop_cols if c in cat_train.columns]
        if cat_drop:
            cat_train = cat_train.drop(columns=cat_drop)
            cat_test = cat_test.drop(columns=cat_drop)

    corr_sample = config.correlation_sample or len(train_num)
    sample = train_num.sample(min(corr_sample, len(train_num)), random_state=config.seed)
    corr = sample.corr(numeric_only=True).abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    high_corr = [col for col in upper.columns if any(upper[col] > 0.985)]
    if high_corr:
        train_num = train_num.drop(columns=high_corr)
        test_num = test_num.drop(columns=high_corr)
        cat_drop = [c for c in high_corr if c in cat_train.columns]
        if cat_drop:
            cat_train = cat_train.drop(columns=cat_drop)
            cat_test = cat_test.drop(columns=cat_drop)
        cat_feature_names = [c for c in cat_feature_names if c in cat_train.columns]

    cat_y = cat_train["TARGET"].astype("int8").copy()
    cat_train = cat_train.drop(columns=["TARGET"])
    cat_train, cat_test = cat_train.align(cat_test, join="inner", axis=1)
    cat_feature_names = [c for c in cat_feature_names if c in cat_train.columns]
    return {
        "train": train_num,
        "test": test_num,
        "target": target,
        "train_ids": train_ids,
        "test_ids": test_ids,
        "cat_train": cat_train,
        "cat_test": cat_test,
        "cat_y": cat_y,
        "cat_features": cat_feature_names,
        "encoders": encoders,
        "dropped": {"zero_variance": drop_cols, "high_corr": high_corr},
    }


def get_importances(X, y, shuffle=False, seed=0):
    if shuffle:
        y = y.sample(frac=1, random_state=seed).reset_index(drop=True)
    fold_imp = np.zeros(X.shape[1])
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    for ti, vi in skf.split(X, y):
        m = lgb.LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=40,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.3,
            min_child_samples=50,
            random_state=seed,
            n_jobs=-1,
            verbose=-1,
        )
        m.fit(
            X.iloc[ti],
            y.iloc[ti],
            eval_set=[(X.iloc[vi], y.iloc[vi])],
            eval_metric="auc",
            callbacks=[lgb.early_stopping(20, verbose=False)],
        )
        fold_imp += m.feature_importances_
    return fold_imp / 3


def null_importance_prune(train, test, target, config: TrainingConfig):
    feat_names = list(train.columns)
    sample_size = min(config.null_importance_sample or len(train), len(train))
    ni_idx = train.sample(sample_size, random_state=config.seed).index
    X_ni = train.loc[ni_idx].reset_index(drop=True)
    y_ni = target.loc[ni_idx].reset_index(drop=True)
    actual_imp = get_importances(X_ni, y_ni, shuffle=False, seed=config.seed)
    null_imps = np.zeros((X_ni.shape[1], config.null_importance_runs))
    for i in range(config.null_importance_runs):
        print(f"  Null run {i + 1}/{config.null_importance_runs}...")
        null_imps[:, i] = get_importances(X_ni, y_ni, shuffle=True, seed=100 + i)
    score_vs_null = actual_imp / (np.percentile(null_imps, 80, axis=1) + 1)
    drop_null = [feat_names[j] for j in range(len(feat_names)) if score_vs_null[j] < 1.0]
    if drop_null:
        train = train.drop(columns=drop_null)
        test = test.drop(columns=drop_null)
    return train, test, drop_null


def feature_subsets(train, target, config: TrainingConfig):
    feat_names = list(train.columns)
    m_imp = lgb.LGBMClassifier(
        n_estimators=2000,
        learning_rate=0.02,
        num_leaves=48,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.3,
        min_child_samples=50,
        random_state=config.seed,
        n_jobs=-1,
        verbose=-1,
    )
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=config.seed)
    imp_arr = np.zeros(len(feat_names))
    for ti, vi in skf.split(train, target):
        m_imp.fit(
            train.iloc[ti],
            target.iloc[ti],
            eval_set=[(train.iloc[vi], target.iloc[vi])],
            eval_metric="auc",
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )
        imp_arr += m_imp.feature_importances_
    imp_rank = pd.DataFrame({"feature": feat_names, "imp": imp_arr / 3}).sort_values("imp", ascending=False)
    return {
        "all": feat_names,
        "top": imp_rank.head(min(config.top_feature_count, len(feat_names)))["feature"].tolist(),
        "no_gp": [f for f in feat_names if not f.startswith("GP")],
        "importance": imp_rank,
    }


def fit_lgb_cv(train, test, target, params, config: TrainingConfig, seed=None, features=None):
    features = features or list(train.columns)
    x_train = train[features]
    x_test = test[features]
    skf = StratifiedKFold(n_splits=config.n_folds, shuffle=True, random_state=seed or config.seed)
    oof = np.zeros(len(x_train))
    pred = np.zeros(len(x_test))
    models = []
    for fold, (ti, vi) in enumerate(skf.split(x_train, target)):
        print(f"\n--- LGB Fold {fold + 1} ---")
        m = lgb.LGBMClassifier(**params)
        m.fit(
            x_train.iloc[ti],
            target.iloc[ti],
            eval_set=[(x_train.iloc[vi], target.iloc[vi])],
            eval_metric="auc",
            callbacks=[lgb.early_stopping(200, verbose=True), lgb.log_evaluation(500)],
        )
        oof[vi] = m.predict_proba(x_train.iloc[vi])[:, 1]
        pred += m.predict_proba(x_test)[:, 1] / config.n_folds
        models.append(m)
    return oof, pred, models, features


def train_models(matrices: dict[str, Any], config: TrainingConfig):
    import xgboost as xgb
    from catboost import CatBoostClassifier, Pool

    train = matrices["train"]
    test = matrices["test"]
    target = matrices["target"]
    train, test, drop_null = null_importance_prune(train, test, target, config)
    subsets = feature_subsets(train, target, config)
    base_params = dict(objective="binary", metric="auc", boosting_type="gbdt", n_estimators=10000, n_jobs=-1, verbose=-1)
    lgb_a_params = dict(base_params, learning_rate=0.01, num_leaves=48, max_depth=7, subsample=0.8, colsample_bytree=0.25, reg_alpha=0.05, reg_lambda=0.1, min_child_samples=40, min_child_weight=30, random_state=42)
    lgb_b_params = dict(base_params, learning_rate=0.008, num_leaves=34, max_depth=5, subsample=0.75, colsample_bytree=0.35, reg_alpha=0.05, reg_lambda=0.2, min_child_samples=60, min_child_weight=50, random_state=123)
    oof_lgb1, test_lgb1, lgb1_models, lgb1_features = fit_lgb_cv(train, test, target, lgb_a_params, config, seed=42, features=subsets["all"])
    oof_lgb2, test_lgb2, lgb2_models, lgb2_features = fit_lgb_cv(train, test, target, lgb_b_params, config, seed=123, features=subsets["top"])

    oof_lgb3 = np.zeros(len(train))
    test_lgb3 = np.zeros(len(test))
    lgb3_models = []
    for seed in config.lgb_seed_average_seeds:
        params = dict(base_params, learning_rate=0.01, num_leaves=40, max_depth=6, subsample=0.8, colsample_bytree=0.3, reg_alpha=0.1, reg_lambda=0.15, min_child_samples=50, min_child_weight=40, random_state=seed)
        oof_seed, test_seed, seed_models, _ = fit_lgb_cv(train, test, target, params, config, seed=seed, features=subsets["no_gp"])
        print(f"  Seed {seed} OOF AUC: {roc_auc_score(target, oof_seed):.6f}")
        oof_lgb3 += oof_seed / len(config.lgb_seed_average_seeds)
        test_lgb3 += test_seed / len(config.lgb_seed_average_seeds)
        lgb3_models.extend(seed_models)

    skf = StratifiedKFold(n_splits=config.n_folds, shuffle=True, random_state=config.seed)
    oof_xgb = np.zeros(len(train))
    test_xgb = np.zeros(len(test))
    xgb_models = []
    for fold, (ti, vi) in enumerate(skf.split(train, target)):
        m = xgb.XGBClassifier(objective="binary:logistic", eval_metric="auc", n_estimators=10000, learning_rate=0.01, max_depth=5, subsample=0.8, colsample_bytree=0.3, reg_alpha=0.1, reg_lambda=1.0, min_child_weight=40, gamma=0.1, random_state=42, n_jobs=-1, verbosity=0, early_stopping_rounds=200, tree_method="hist")
        m.fit(train.iloc[ti], target.iloc[ti], eval_set=[(train.iloc[vi], target.iloc[vi])], verbose=500)
        oof_xgb[vi] = m.predict_proba(train.iloc[vi])[:, 1]
        test_xgb += m.predict_proba(test)[:, 1] / config.n_folds
        xgb_models.append(m)

    cat_train = matrices["cat_train"].reindex(columns=train.columns.intersection(matrices["cat_train"].columns), fill_value=np.nan)
    cat_test = matrices["cat_test"].reindex(columns=cat_train.columns, fill_value=np.nan)
    cat_features = [c for c in matrices["cat_features"] if c in cat_train.columns]
    cat_feature_indices = [cat_train.columns.get_loc(c) for c in cat_features]
    oof_cat = np.zeros(len(cat_train))
    test_cat = np.zeros(len(cat_test))
    cat_models = []
    for fold, (ti, vi) in enumerate(skf.split(cat_train, matrices["cat_y"])):
        params = dict(loss_function="Logloss", eval_metric="AUC", iterations=10000, learning_rate=0.03, depth=7, l2_leaf_reg=3.0, random_seed=42 + fold, verbose=500, early_stopping_rounds=300, task_type="CPU", bootstrap_type="Bernoulli", subsample=0.8, grow_policy="SymmetricTree", leaf_estimation_iterations=3, rsm=0.3)
        train_pool = Pool(cat_train.iloc[ti], label=matrices["cat_y"].iloc[ti], cat_features=cat_feature_indices)
        valid_pool = Pool(cat_train.iloc[vi], label=matrices["cat_y"].iloc[vi], cat_features=cat_feature_indices)
        test_pool = Pool(cat_test, cat_features=cat_feature_indices)
        m = CatBoostClassifier(**params)
        m.fit(train_pool, eval_set=valid_pool, use_best_model=True)
        oof_cat[vi] = m.predict_proba(valid_pool)[:, 1]
        test_cat += m.predict_proba(test_pool)[:, 1] / config.n_folds
        cat_models.append(m)

    models = {
        "lgb_a": (oof_lgb1, test_lgb1),
        "lgb_b": (oof_lgb2, test_lgb2),
        "lgb_seed": (oof_lgb3, test_lgb3),
        "xgb": (oof_xgb, test_xgb),
        "cat": (oof_cat, test_cat),
    }
    names = list(models.keys())
    oof_stack = np.column_stack([models[n][0] for n in names])
    test_stack = np.column_stack([models[n][1] for n in names])
    stack_models = []
    oof_stack_lr = np.zeros(len(train))
    test_stack_lr = np.zeros(len(test))
    for ti, vi in StratifiedKFold(n_splits=config.n_folds, shuffle=True, random_state=789).split(oof_stack, target):
        lr = LogisticRegression(C=0.35, max_iter=2000, solver="lbfgs", random_state=789)
        lr.fit(oof_stack[ti], target.values[ti])
        oof_stack_lr[vi] = lr.predict_proba(oof_stack[vi])[:, 1]
        test_stack_lr += lr.predict_proba(test_stack)[:, 1] / config.n_folds
        stack_models.append(lr)

    blend_oof = [rank_norm(models[n][0]) for n in names] + [rank_norm(oof_stack_lr)]
    blend_test = [rank_norm(models[n][1]) for n in names] + [rank_norm(test_stack_lr)]
    blend_names = names + ["stack_lr"]

    def neg_auc(w):
        w = np.clip(w, 0, 1)
        w = w / (w.sum() + 1e-12)
        return -roc_auc_score(target, sum(wi * ri for wi, ri in zip(w, blend_oof)))

    result = minimize(neg_auc, np.ones(len(blend_names)) / len(blend_names), method="SLSQP", bounds=[(0.0, 0.60)] * len(blend_names), constraints=({"type": "eq", "fun": lambda w: np.sum(np.clip(w, 0, 1)) - 1.0},), options={"maxiter": 3000, "ftol": 1e-10})
    best_w = np.clip(result.x, 0, 1)
    best_w = best_w / (best_w.sum() + 1e-12)
    blend_auc = -result.fun
    test_pred = sum(w * r for w, r in zip(best_w, blend_test))

    metrics = {f"{name}_auc": float(roc_auc_score(target, oof)) for name, (oof, _) in models.items()}
    metrics["stack_lr_auc"] = float(roc_auc_score(target, oof_stack_lr))
    metrics["blend_auc"] = float(blend_auc)
    artifact = {
        "features": list(train.columns),
        "cat_features": cat_features,
        "blend_names": blend_names,
        "blend_weights": best_w.tolist(),
        "models": {
            "lgb_a": {"models": lgb1_models, "features": lgb1_features},
            "lgb_b": {"models": lgb2_models, "features": lgb2_features},
            "lgb_seed": {"models": lgb3_models, "features": subsets["no_gp"], "n_seed_groups": len(config.lgb_seed_average_seeds)},
            "xgb": {"models": xgb_models, "features": list(train.columns)},
            "cat": {"models": cat_models, "features": list(cat_train.columns), "cat_features": cat_features},
            "stack_lr": {"models": stack_models, "base_names": names},
        },
        "metrics": metrics,
        "drop_null": drop_null,
    }
    return artifact, test_pred, metrics


def setup_mlflow(config: TrainingConfig):
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI") or config.mlflow_tracking_uri
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    owner = os.getenv("DAGSHUB_REPO_OWNER") or config.dagshub_repo_owner
    repo = os.getenv("DAGSHUB_REPO_NAME") or config.dagshub_repo_name
    token = os.getenv("DAGSHUB_TOKEN")
    if owner and repo and token and not tracking_uri:
        mlflow.set_tracking_uri(f"https://dagshub.com/{owner}/{repo}.mlflow")
    mlflow.set_experiment(config.experiment_name)


def train_pipeline(params_path: str | Path = "params.yaml", raw_data_dir: Path = RAW_DATA_DIR):
    config = load_training_config(params_path)
    config.output_submission.parent.mkdir(parents=True, exist_ok=True)
    config.model_artifact.parent.mkdir(parents=True, exist_ok=True)
    config.metrics_path.parent.mkdir(parents=True, exist_ok=True)
    setup_mlflow(config)
    with mlflow.start_run():
        mlflow.log_params({k: v for k, v in asdict(config).items() if isinstance(v, (str, int, float, bool, type(None)))})
        for table, limit in config.row_limits.items():
            mlflow.log_param(f"row_limit_{table}", limit)
        train, test = build_feature_frame(config, raw_data_dir)
        matrices = prepare_model_matrices(train, test, config)
        artifact, test_pred, metrics = train_models(matrices, config)
        submission = pd.DataFrame({"SK_ID_CURR": matrices["test_ids"].astype(int), "TARGET": test_pred})
        submission.to_csv(config.output_submission, index=False)
        joblib.dump(artifact, config.model_artifact)
        with config.metrics_path.open("w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        with config.feature_metadata.open("w", encoding="utf-8") as f:
            json.dump({"features": artifact["features"], "cat_features": artifact["cat_features"]}, f, indent=2)
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(str(config.output_submission))
        mlflow.log_artifact(str(config.metrics_path))
        mlflow.log_artifact(str(config.feature_metadata))
        mlflow.sklearn.log_model(artifact["models"]["stack_lr"]["models"][0], "stack_lr_model")
        mlflow.log_artifact(str(config.model_artifact))
    print(f"Saved model artifact: {config.model_artifact}")
    print(f"Saved submission: {config.output_submission}")
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--params", default="params.yaml")
    args = parser.parse_args()
    metrics = train_pipeline(args.params)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
