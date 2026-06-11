from __future__ import annotations

import gc
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

from src.config import RAW_DATA_DIR, TrainingConfig
from src.data import read_csv_optimized, read_table, reduce_memory_usage, table_path


def time_weighted_agg(df, group_col, value_cols, time_col, prefix, decay=0.002):
    ids = df[group_col].unique()
    result = pd.DataFrame({group_col: ids})
    w = np.exp(decay * df[time_col].values.astype("float64"))
    for vc in value_cols:
        mask = df[vc].notna().values
        if mask.sum() == 0:
            result[f"{prefix}_{vc}_TWMEAN"] = np.nan
            continue
        temp = pd.DataFrame(
            {
                group_col: df[group_col].values[mask],
                "_wv": df[vc].values[mask].astype("float64") * w[mask],
                "_w": w[mask],
            }
        )
        agg = temp.groupby(group_col)[["_wv", "_w"]].sum()
        agg[f"{prefix}_{vc}_TWMEAN"] = (agg["_wv"] / agg["_w"]).astype("float32")
        result = result.merge(agg[[f"{prefix}_{vc}_TWMEAN"]].reset_index(), on=group_col, how="left")
    return result


def compute_trend(df, group_col, value_col, time_col, prefix):
    temp = df[[group_col, value_col, time_col]].dropna().copy()
    out_col = f"{prefix}_{value_col}_TREND"
    if len(temp) == 0:
        return pd.DataFrame({group_col: df[group_col].unique(), out_col: np.nan})
    gcounts = temp.groupby(group_col)[value_col].transform("count")
    temp = temp[gcounts >= 3].copy()
    if len(temp) == 0:
        return pd.DataFrame({group_col: df[group_col].unique(), out_col: np.nan})
    g = temp.groupby(group_col)
    dt = temp[time_col].astype("float64") - g[time_col].transform("mean").astype("float64")
    dv = temp[value_col].astype("float64") - g[value_col].transform("mean").astype("float64")
    temp["_dtdv"] = dt * dv
    temp["_dt2"] = dt**2
    agg = temp.groupby(group_col)[["_dtdv", "_dt2"]].sum()
    agg[out_col] = (agg["_dtdv"] / (agg["_dt2"] + 1e-8)).astype("float32")
    return agg[[out_col]].reset_index()


def agg_time_window(df, group_col, cols, time_col, cutoff, prefix):
    sub = df[df[time_col] >= cutoff]
    if len(sub) == 0:
        return pd.DataFrame({group_col: df[group_col].unique()})
    valid_cols = [c for c in cols if c in sub.columns]
    if not valid_cols:
        return pd.DataFrame({group_col: df[group_col].unique()})
    agg = sub.groupby(group_col)[valid_cols].agg(["mean", "max", "sum"])
    agg.columns = [f"{prefix}_{c[0]}_{c[1].upper()}" for c in agg.columns]
    cnt = sub.groupby(group_col).size().reset_index(name=f"{prefix}_COUNT")
    return agg.reset_index().merge(cnt, on=group_col, how="left")


def add_frequency_features(train_df, test_df, cols):
    full = pd.concat([train_df.drop(columns=["TARGET"], errors="ignore"), test_df], axis=0, ignore_index=True)
    for col in cols:
        if col not in full.columns:
            continue
        vc = full[col].fillna("__nan__").value_counts(dropna=False)
        full[f"{col}_FREQ"] = full[col].fillna("__nan__").map(vc).astype("float32")
        full[f"{col}_FREQ_NORM"] = (full[f"{col}_FREQ"] / len(full)).astype("float32")
    out_train = full.iloc[: len(train_df)].copy()
    out_test = full.iloc[len(train_df) :].copy()
    if "TARGET" in train_df.columns:
        out_train["TARGET"] = train_df["TARGET"].values
    return out_train, out_test


def add_groupby_ratio_features(train_df, test_df, cat_cols, num_cols):
    full = pd.concat([train_df.drop(columns=["TARGET"], errors="ignore"), test_df], axis=0, ignore_index=True)
    for cat in cat_cols:
        if cat not in full.columns:
            continue
        for num in num_cols:
            if num not in full.columns:
                continue
            gp = full.groupby(cat)[num].agg(["mean", "median", "std"]).reset_index()
            gp.columns = [cat, f"GB_{cat}_{num}_MEAN", f"GB_{cat}_{num}_MEDIAN", f"GB_{cat}_{num}_STD"]
            full = full.merge(gp, on=cat, how="left")
            full[f"GB_{cat}_{num}_DIFF"] = full[num] - full[f"GB_{cat}_{num}_MEAN"]
            full[f"GB_{cat}_{num}_RATIO"] = full[num] / (full[f"GB_{cat}_{num}_MEAN"] + 1e-6)
    full = reduce_memory_usage(full)
    out_train = full.iloc[: len(train_df)].copy()
    out_test = full.iloc[len(train_df) :].copy()
    if "TARGET" in train_df.columns:
        out_train["TARGET"] = train_df["TARGET"].values
    return out_train, out_test


def add_target_encoding(train_df, test_df, target_col, cols, n_splits=5, smoothing=40, min_samples_leaf=80, seed=42):
    train_df = train_df.copy()
    test_df = test_df.copy()
    global_mean = train_df[target_col].mean()
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for col in cols:
        if col not in train_df.columns:
            continue
        new_col = f"{col}_TE"
        if new_col in train_df.columns:
            continue
        tr_enc = np.zeros(len(train_df), dtype="float32")
        for tr_idx, va_idx in skf.split(train_df, train_df[target_col]):
            stats = train_df.iloc[tr_idx].groupby(col)[target_col].agg(["mean", "count"])
            smooth = (stats["count"] * stats["mean"] + smoothing * global_mean) / (stats["count"] + smoothing)
            smooth[stats["count"] < min_samples_leaf] = global_mean
            tr_enc[va_idx] = train_df.iloc[va_idx][col].map(smooth).fillna(global_mean).values.astype("float32")
        full_stats = train_df.groupby(col)[target_col].agg(["mean", "count"])
        full_smooth = (
            full_stats["count"] * full_stats["mean"] + smoothing * global_mean
        ) / (full_stats["count"] + smoothing)
        full_smooth[full_stats["count"] < min_samples_leaf] = global_mean
        train_df[new_col] = tr_enc
        test_df[new_col] = test_df[col].map(full_smooth).fillna(global_mean).values.astype("float32")
    return train_df, test_df


def _row_sum_numeric(frame, cols, dtype="float32"):
    cols = [c for c in cols if c in frame.columns]
    if not cols:
        return pd.Series(np.zeros(len(frame), dtype=dtype), index=frame.index)
    return frame[cols].apply(pd.to_numeric, errors="coerce").sum(axis=1).astype(dtype)


def application_features(df):
    out = df.copy()
    out["DAYS_EMPLOYED"] = out["DAYS_EMPLOYED"].replace(365243, np.nan)
    out["DAYS_EMPLOYED_ANOM"] = (df["DAYS_EMPLOYED"] == 365243).astype("int8")
    ext = ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]
    out["EXT_MEAN"] = out[ext].mean(axis=1)
    out["EXT_STD"] = out[ext].std(axis=1)
    out["EXT_PROD"] = out["EXT_SOURCE_1"] * out["EXT_SOURCE_2"] * out["EXT_SOURCE_3"]
    out["EXT_MIN"] = out[ext].min(axis=1)
    out["EXT_MAX"] = out[ext].max(axis=1)
    out["EXT_NANCOUNT"] = out[ext].isna().sum(axis=1)
    out["EXT_S1xS2"] = out["EXT_SOURCE_1"] * out["EXT_SOURCE_2"]
    out["EXT_S1xS3"] = out["EXT_SOURCE_1"] * out["EXT_SOURCE_3"]
    out["EXT_S2xS3"] = out["EXT_SOURCE_2"] * out["EXT_SOURCE_3"]
    out["EXT_S2divS3"] = out["EXT_SOURCE_2"] / (out["EXT_SOURCE_3"] + 1e-4)
    out["EXT_S1divS2"] = out["EXT_SOURCE_1"] / (out["EXT_SOURCE_2"] + 1e-4)
    for i in [1, 2, 3]:
        col = f"EXT_SOURCE_{i}"
        out[f"{col}_SQ"] = out[col] ** 2
        out[f"{col}_CB"] = out[col] ** 3
    out["EXT_S2xBIRTH"] = out["EXT_SOURCE_2"] * out["DAYS_BIRTH"]
    out["EXT_S1xBIRTH"] = out["EXT_SOURCE_1"] * out["DAYS_BIRTH"]
    out["EXT_S3xBIRTH"] = out["EXT_SOURCE_3"] * out["DAYS_BIRTH"]
    out["EXT_S2xEMPL"] = out["EXT_SOURCE_2"] * out["DAYS_EMPLOYED"]
    out["EXT_S3xEMPL"] = out["EXT_SOURCE_3"] * out["DAYS_EMPLOYED"]
    out["GP1"] = out["EXT_SOURCE_2"] ** 2 * out["EXT_SOURCE_3"]
    out["GP2"] = out["EXT_SOURCE_1"] * out["DAYS_BIRTH"] / (out["AMT_ANNUITY"] + 1)
    out["GP3"] = out["EXT_SOURCE_2"] * out["REGION_RATING_CLIENT_W_CITY"]
    out["GP4"] = out["EXT_SOURCE_3"] * np.log1p(np.abs(out["DAYS_BIRTH"]))
    out["GP5"] = out["AMT_ANNUITY"] * out["EXT_SOURCE_3"] / (out["AMT_INCOME_TOTAL"] + 1)
    out["GP6"] = out["EXT_SOURCE_1"] * out["DAYS_ID_PUBLISH"] / (out["DAYS_BIRTH"] + 1)
    out["GP7"] = out["EXT_SOURCE_2"] * out["AMT_CREDIT"] / (out["AMT_GOODS_PRICE"] + 1)
    out["GP8"] = out["EXT_SOURCE_1"] * out["EXT_SOURCE_2"] * out["EXT_SOURCE_3"] / (out["AMT_CREDIT"] + 1)
    out["GP9"] = out["EXT_MEAN"] * out["DAYS_EMPLOYED"] / (out["DAYS_BIRTH"] - 1)
    out["GP10"] = (out["AMT_GOODS_PRICE"] - out["AMT_CREDIT"]) * out["EXT_SOURCE_2"] / (out["AMT_ANNUITY"] + 1)
    out["CREDIT_INCOME_RATIO"] = out["AMT_CREDIT"] / (out["AMT_INCOME_TOTAL"] + 1)
    out["ANNUITY_INCOME_RATIO"] = out["AMT_ANNUITY"] / (out["AMT_INCOME_TOTAL"] + 1)
    out["CREDIT_ANNUITY_RATIO"] = out["AMT_CREDIT"] / (out["AMT_ANNUITY"] + 1)
    out["CREDIT_GOODS_RATIO"] = out["AMT_CREDIT"] / (out["AMT_GOODS_PRICE"] + 1)
    out["GOODS_INCOME_RATIO"] = out["AMT_GOODS_PRICE"] / (out["AMT_INCOME_TOTAL"] + 1)
    out["INCOME_PER_CHILD"] = out["AMT_INCOME_TOTAL"] / (out["CNT_CHILDREN"] + 1)
    out["INCOME_PER_FAM"] = out["AMT_INCOME_TOTAL"] / (out["CNT_FAM_MEMBERS"] + 1)
    out["ANNUITY_CREDIT_RATIO"] = out["AMT_ANNUITY"] / (out["AMT_CREDIT"] + 1)
    out["PAYMENT_LENGTH"] = out["AMT_CREDIT"] / (out["AMT_ANNUITY"] + 1)
    out["DOWN_PAYMENT"] = out["AMT_GOODS_PRICE"] - out["AMT_CREDIT"]
    out["DOWN_PAYMENT_RATIO"] = out["DOWN_PAYMENT"] / (out["AMT_GOODS_PRICE"] + 1)
    out["DAYS_BIRTH_YRS"] = out["DAYS_BIRTH"] / -365.25
    out["DAYS_EMPLOYED_YRS"] = out["DAYS_EMPLOYED"] / -365.25
    out["EMPLOYED_TO_BIRTH"] = out["DAYS_EMPLOYED"] / (out["DAYS_BIRTH"] + 1)
    out["CAR_AGE_TO_BIRTH"] = out["OWN_CAR_AGE"] / (out["DAYS_BIRTH_YRS"] + 1)
    out["ID_PUBLISH_TO_BIRTH"] = out["DAYS_ID_PUBLISH"] / (out["DAYS_BIRTH"] + 1)
    out["PHONE_TO_BIRTH"] = out["DAYS_LAST_PHONE_CHANGE"] / (out["DAYS_BIRTH"] + 1)
    out["PHONE_TO_EMPLOYED"] = out["DAYS_LAST_PHONE_CHANGE"] / (out["DAYS_EMPLOYED"] + 1)
    out["REG_TO_BIRTH"] = out["DAYS_REGISTRATION"] / (out["DAYS_BIRTH"] + 1)
    out["AGE_RANGE"] = pd.cut(
        out["DAYS_BIRTH_YRS"], bins=[0, 25, 30, 35, 40, 45, 50, 55, 60, 65, 100], labels=False
    )
    out["INCOME_EMPLOYED"] = out["AMT_INCOME_TOTAL"] * out["DAYS_EMPLOYED_YRS"]
    out["DOCUMENT_COUNT"] = _row_sum_numeric(out, [c for c in out.columns if "FLAG_DOCUMENT" in c])
    out["DEF_30_RATIO"] = out["DEF_30_CNT_SOCIAL_CIRCLE"] / (out["OBS_30_CNT_SOCIAL_CIRCLE"] + 1)
    out["DEF_60_RATIO"] = out["DEF_60_CNT_SOCIAL_CIRCLE"] / (out["OBS_60_CNT_SOCIAL_CIRCLE"] + 1)
    out["APP_NULLS"] = out.isna().sum(axis=1).astype("int16")
    out["CITY_RATING_x_EXT2"] = out["REGION_RATING_CLIENT_W_CITY"] * out["EXT_SOURCE_2"]
    out["EMPLOYED_TO_ID"] = out["DAYS_EMPLOYED"] / (out["DAYS_ID_PUBLISH"] + 1)
    out["ID_TO_BIRTH_RATIO"] = out["DAYS_ID_PUBLISH"] / (out["DAYS_BIRTH"] + 1)
    out["REG_TO_EMPLOYED_RATIO"] = out["DAYS_REGISTRATION"] / (out["DAYS_EMPLOYED"] + 1)
    out["CREDIT_PER_PERSON"] = out["AMT_CREDIT"] / (out["CNT_FAM_MEMBERS"] + 1)
    out["ANNUITY_PER_PERSON"] = out["AMT_ANNUITY"] / (out["CNT_FAM_MEMBERS"] + 1)
    out["INCOME_CREDIT_PERC"] = out["AMT_INCOME_TOTAL"] / (out["AMT_CREDIT"] + 1)
    out["INCOME_ANNUITY_PERC"] = out["AMT_INCOME_TOTAL"] / (out["AMT_ANNUITY"] + 1)
    out["EXT_RANGE"] = out["EXT_MAX"] - out["EXT_MIN"]
    out["EXT_SOURCE_SPREAD"] = out["EXT_STD"] / (out["EXT_MEAN"] + 1e-4)
    out["PHONE_MINUS_REG"] = out["DAYS_LAST_PHONE_CHANGE"] - out["DAYS_REGISTRATION"]
    out["CAR_EMPLOYED_RATIO"] = out["OWN_CAR_AGE"] / (out["DAYS_EMPLOYED_YRS"] + 1)
    out["CHILDREN_RATIO"] = out["CNT_CHILDREN"] / (out["CNT_FAM_MEMBERS"] + 1)
    out["OBS_30_60_RATIO"] = out["OBS_30_CNT_SOCIAL_CIRCLE"] / (out["OBS_60_CNT_SOCIAL_CIRCLE"] + 1)
    out["DEF_30_60_RATIO"] = out["DEF_30_CNT_SOCIAL_CIRCLE"] / (out["DEF_60_CNT_SOCIAL_CIRCLE"] + 1)
    out["AMT_REQ_SUM"] = _row_sum_numeric(out, [c for c in out.columns if c.startswith("AMT_REQ_CREDIT_BUREAU_")])
    contact_cols = [
        c
        for c in ["FLAG_MOBIL", "FLAG_EMP_PHONE", "FLAG_WORK_PHONE", "FLAG_CONT_MOBILE", "FLAG_PHONE", "FLAG_EMAIL"]
        if c in out.columns
    ]
    if contact_cols:
        out["FLAG_CONTACTS_SUM"] = _row_sum_numeric(out, contact_cols)
    out["CREDIT_TERM"] = out["AMT_ANNUITY"] / (out["AMT_CREDIT"] + 1)
    out["DAYS_EMPLOYED_PERC"] = out["DAYS_EMPLOYED"] / (out["DAYS_BIRTH"] + 1)
    out["INCOME_CREDIT_PERC2"] = out["AMT_INCOME_TOTAL"] / (out["AMT_CREDIT"] + 1)
    out["EXT_WEIGHTED"] = 2 * out["EXT_SOURCE_2"] + out["EXT_SOURCE_3"] + 0.5 * out["EXT_SOURCE_1"]
    out["REGION_POP_x_EXT"] = out["REGION_POPULATION_RELATIVE"] * out["EXT_MEAN"]
    out["HOUR_APPR_x_EXT2"] = out["HOUR_APPR_PROCESS_START"] * out["EXT_SOURCE_2"]
    out["LIVE_REGION_DIFF"] = (
        out["REG_REGION_NOT_LIVE_REGION"].astype(float)
        + out["REG_REGION_NOT_WORK_REGION"].astype(float)
        + out.get("LIVE_REGION_NOT_WORK_REGION", pd.Series(0, index=out.index)).astype(float)
    )
    return out


def bureau_and_balance_features(config: TrainingConfig, raw_data_dir: Path = RAW_DATA_DIR):
    bureau = read_table("bureau", config, raw_data_dir)
    bb = read_table("bureau_balance", config, raw_data_dir)
    bb_counts = bb.pivot_table(index="SK_ID_BUREAU", columns="STATUS", values="MONTHS_BALANCE", aggfunc="count", fill_value=0)
    bb_counts.columns = [f"BB_STATUS_{c}" for c in bb_counts.columns]
    bb_months = bb.groupby("SK_ID_BUREAU")["MONTHS_BALANCE"].agg(BB_MONTHS_MIN="min", BB_MONTHS_MAX="max", BB_MONTHS_SIZE="size").reset_index()
    bureau = bureau.merge(bb_months.merge(bb_counts.reset_index(), on="SK_ID_BUREAU", how="left"), on="SK_ID_BUREAU", how="left")
    del bb, bb_counts, bb_months
    bureau["CREDIT_DURATION"] = bureau["DAYS_CREDIT_ENDDATE"] - bureau["DAYS_CREDIT"]
    bureau["ENDDATE_DIFF"] = bureau["DAYS_CREDIT_ENDDATE"] - bureau["DAYS_ENDDATE_FACT"]
    bureau["DEBT_CREDIT_RATIO"] = bureau["AMT_CREDIT_SUM_DEBT"] / (bureau["AMT_CREDIT_SUM"] + 1)
    bureau["OVERDUE_DEBT_RATIO"] = bureau["AMT_CREDIT_SUM_OVERDUE"] / (bureau["AMT_CREDIT_SUM_DEBT"] + 1)
    bureau["AMT_ANNUITY_CREDIT"] = bureau["AMT_ANNUITY"] / (bureau["AMT_CREDIT_SUM"] + 1)
    bureau["CREDIT_OVERDUE_RATIO"] = bureau["AMT_CREDIT_SUM_OVERDUE"] / (bureau["AMT_CREDIT_SUM"] + 1)
    bureau["DAYS_CREDIT_UPDATE_DIFF"] = bureau["DAYS_CREDIT_UPDATE"] - bureau["DAYS_CREDIT"]
    feat = _base_group_agg(bureau, "BURO", ["SK_ID_BUREAU", "SK_ID_CURR"])
    for status in ["Active", "Closed"]:
        sub = bureau[bureau["CREDIT_ACTIVE"] == status]
        if len(sub) > 0:
            key = [c for c in ["AMT_CREDIT_SUM", "AMT_CREDIT_SUM_DEBT", "DAYS_CREDIT", "DAYS_CREDIT_ENDDATE", "DEBT_CREDIT_RATIO"] if c in sub.columns]
            sa = sub.groupby("SK_ID_CURR")[key].agg(["mean", "sum", "max", "min"])
            sa.columns = [f"BURO_{status.upper()}_{c[0]}_{c[1].upper()}" for c in sa.columns]
            feat = feat.merge(sa.reset_index().merge(sub.groupby("SK_ID_CURR").size().reset_index(name=f"BURO_{status.upper()}_COUNT"), on="SK_ID_CURR"), on="SK_ID_CURR", how="left")
    tw_cols = [c for c in ["AMT_CREDIT_SUM", "AMT_CREDIT_SUM_DEBT", "CREDIT_DAY_OVERDUE", "DEBT_CREDIT_RATIO"] if c in bureau.columns]
    for days, label in [(-180, "6M"), (-365, "1Y"), (-730, "2Y"), (-1095, "3Y"), (-1825, "5Y")]:
        feat = feat.merge(agg_time_window(bureau, "SK_ID_CURR", tw_cols, "DAYS_CREDIT", days, f"BURO_{label}"), on="SK_ID_CURR", how="left")
    feat = feat.merge(time_weighted_agg(bureau, "SK_ID_CURR", ["AMT_CREDIT_SUM", "AMT_CREDIT_SUM_DEBT", "DEBT_CREDIT_RATIO"], "DAYS_CREDIT", "BURO", decay=0.001), on="SK_ID_CURR", how="left")
    for col in ["AMT_CREDIT_SUM_DEBT", "DEBT_CREDIT_RATIO"]:
        feat = feat.merge(compute_trend(bureau, "SK_ID_CURR", col, "DAYS_CREDIT", "BURO"), on="SK_ID_CURR", how="left")
    last = bureau.sort_values("DAYS_CREDIT", ascending=False).groupby("SK_ID_CURR").first().reset_index()
    for col in ["DAYS_CREDIT", "AMT_CREDIT_SUM", "AMT_CREDIT_SUM_DEBT", "DEBT_CREDIT_RATIO", "CREDIT_DAY_OVERDUE"]:
        if col in last.columns:
            feat = feat.merge(last[["SK_ID_CURR", col]].rename(columns={col: f"BURO_LAST_{col}"}), on="SK_ID_CURR", how="left")
    feat = feat.merge(bureau.groupby("SK_ID_CURR")["CREDIT_TYPE"].nunique().reset_index(name="BURO_CREDIT_TYPE_NUNIQUE"), on="SK_ID_CURR", how="left")
    feat = feat.merge((bureau.groupby("SK_ID_CURR")["CREDIT_DAY_OVERDUE"].max() > 0).astype("int8").reset_index(name="BURO_OVERDUE_EVER"), on="SK_ID_CURR", how="left")
    return feat


def _base_group_agg(df, prefix, exclude):
    num_cols = [c for c in df.columns if df[c].dtype != "object" and c not in exclude]
    out = df.groupby("SK_ID_CURR")[num_cols].agg(["min", "max", "mean", "sum", "var"])
    out.columns = [f"{prefix}_{c[0]}_{c[1].upper()}" for c in out.columns]
    feat = out.reset_index()
    cat_cols = [c for c in df.columns if df[c].dtype == "object"]
    if cat_cols:
        cat = pd.get_dummies(df[["SK_ID_CURR"] + cat_cols], columns=cat_cols, dummy_na=True)
        cat = cat.groupby("SK_ID_CURR").mean().reset_index()
        cat.columns = ["SK_ID_CURR"] + [f"{prefix}_{c}" for c in cat.columns if c != "SK_ID_CURR"]
        feat = feat.merge(cat, on="SK_ID_CURR", how="left")
    return feat.merge(df.groupby("SK_ID_CURR").size().reset_index(name=f"{prefix}_COUNT"), on="SK_ID_CURR", how="left")


def previous_application_features(config: TrainingConfig, raw_data_dir: Path = RAW_DATA_DIR):
    prev = read_table("previous_application", config, raw_data_dir)
    for col in [c for c in prev.columns if "DAYS_" in c]:
        prev[col] = prev[col].replace(365243, np.nan)
    prev["APP_CREDIT_RATIO"] = prev["AMT_APPLICATION"] / (prev["AMT_CREDIT"] + 1)
    prev["CREDIT_GOODS_P"] = prev["AMT_CREDIT"] / (prev["AMT_GOODS_PRICE"] + 1)
    prev["APP_GOODS_RATIO"] = prev["AMT_APPLICATION"] / (prev["AMT_GOODS_PRICE"] + 1)
    prev["DAYS_FIRST_DUE_DIFF"] = prev["DAYS_FIRST_DUE"] - prev["DAYS_FIRST_DRAWING"]
    prev["DAYS_LAST_DUE_DIFF"] = prev["DAYS_LAST_DUE_1ST_VERSION"] - prev["DAYS_LAST_DUE"]
    prev["DOWN_PAYMENT_P"] = prev["AMT_DOWN_PAYMENT"] / (prev["AMT_CREDIT"] + 1)
    prev["INTEREST_SHARE"] = prev["CNT_PAYMENT"] * prev["AMT_ANNUITY"] - prev["AMT_CREDIT"]
    prev["INTEREST_RATE"] = prev["INTEREST_SHARE"] / (prev["AMT_CREDIT"] + 1)
    feat = _base_group_agg(prev, "PREV", ["SK_ID_CURR", "SK_ID_PREV"])
    for status in ["Approved", "Refused", "Canceled"]:
        sub = prev[prev["NAME_CONTRACT_STATUS"] == status]
        if len(sub) > 0:
            sa = sub.groupby("SK_ID_CURR")[["AMT_CREDIT", "AMT_APPLICATION", "AMT_ANNUITY", "DAYS_DECISION"]].agg(["mean", "max", "min"])
            sa.columns = [f"PREV_{status.upper()}_{c[0]}_{c[1].upper()}" for c in sa.columns]
            feat = feat.merge(sa.reset_index().merge(sub.groupby("SK_ID_CURR").size().reset_index(name=f"PREV_{status.upper()}_COUNT"), on="SK_ID_CURR"), on="SK_ID_CURR", how="left")
    for ctype in ["Cash loans", "Revolving loans"]:
        sub = prev[prev["NAME_CONTRACT_TYPE"] == ctype]
        if len(sub) > 0:
            label = "CASH" if "Cash" in ctype else "REVOLV"
            sa = sub.groupby("SK_ID_CURR")[["AMT_CREDIT", "AMT_ANNUITY", "APP_CREDIT_RATIO"]].agg(["mean", "sum", "max"])
            sa.columns = [f"PREV_{label}_{c[0]}_{c[1].upper()}" for c in sa.columns]
            feat = feat.merge(sa.reset_index().merge(sub.groupby("SK_ID_CURR").size().reset_index(name=f"PREV_{label}_COUNT"), on="SK_ID_CURR"), on="SK_ID_CURR", how="left")
    tw_cols = [c for c in ["AMT_CREDIT", "AMT_ANNUITY", "APP_CREDIT_RATIO", "INTEREST_RATE"] if c in prev.columns]
    for days, label in [(-180, "6M"), (-365, "1Y"), (-730, "2Y"), (-1095, "3Y")]:
        feat = feat.merge(agg_time_window(prev, "SK_ID_CURR", tw_cols, "DAYS_DECISION", days, f"PREV_{label}"), on="SK_ID_CURR", how="left")
    feat = feat.merge(time_weighted_agg(prev, "SK_ID_CURR", ["AMT_CREDIT", "AMT_ANNUITY", "APP_CREDIT_RATIO"], "DAYS_DECISION", "PREV", decay=0.001), on="SK_ID_CURR", how="left")
    app_rate = prev.groupby("SK_ID_CURR")["NAME_CONTRACT_STATUS"].apply(lambda x: (x == "Approved").mean()).reset_index(name="PREV_APPROVAL_RATE")
    feat = feat.merge(app_rate, on="SK_ID_CURR", how="left")
    last = prev.sort_values("DAYS_DECISION", ascending=False).groupby("SK_ID_CURR").first().reset_index()
    for col in ["DAYS_DECISION", "AMT_CREDIT", "APP_CREDIT_RATIO", "INTEREST_RATE"]:
        if col in last.columns:
            feat = feat.merge(last[["SK_ID_CURR", col]].rename(columns={col: f"PREV_LAST_{col}"}), on="SK_ID_CURR", how="left")
    return feat


def pos_cash_features(config: TrainingConfig, raw_data_dir: Path = RAW_DATA_DIR):
    pos = read_table("pos_cash_balance", config, raw_data_dir)
    pos["SK_DPD_RATIO"] = pos["SK_DPD"] / (pos["SK_DPD_DEF"] + 1)
    pos["LATE_POS"] = (pos["SK_DPD"] > 0).astype("int8")
    feat = _base_group_agg(pos, "POS", ["SK_ID_CURR", "SK_ID_PREV"])
    feat = feat.merge(pos.groupby("SK_ID_CURR")["LATE_POS"].mean().reset_index(name="POS_LATE_RATE"), on="SK_ID_CURR", how="left")
    for months, label in [(-3, "3M"), (-6, "6M"), (-12, "12M"), (-24, "24M")]:
        feat = feat.merge(agg_time_window(pos, "SK_ID_CURR", ["SK_DPD", "SK_DPD_DEF", "CNT_INSTALMENT", "CNT_INSTALMENT_FUTURE"], "MONTHS_BALANCE", months, f"POS_{label}"), on="SK_ID_CURR", how="left")
    feat = _merge_per_loan(feat, pos, "POS", {"DPD_MAX": ("SK_DPD", "max"), "DPD_MEAN": ("SK_DPD", "mean"), "LATE_RATE": ("LATE_POS", "mean"), "MONTHS": ("MONTHS_BALANCE", "count")})
    feat = feat.merge(time_weighted_agg(pos, "SK_ID_CURR", ["SK_DPD", "CNT_INSTALMENT_FUTURE"], "MONTHS_BALANCE", "POS", decay=0.02), on="SK_ID_CURR", how="left")
    feat = feat.merge(compute_trend(pos, "SK_ID_CURR", "SK_DPD", "MONTHS_BALANCE", "POS"), on="SK_ID_CURR", how="left")
    if "NAME_CONTRACT_STATUS" in pos.columns:
        completed = pos[pos["NAME_CONTRACT_STATUS"] == "Completed"]
        comp_rate = (completed.groupby("SK_ID_CURR").size() / pos.groupby("SK_ID_CURR").size()).reset_index(name="POS_COMPLETED_RATE")
        feat = feat.merge(comp_rate, on="SK_ID_CURR", how="left")
    return feat


def _merge_per_loan(feat, df, prefix, aggregations):
    named_aggs = {f"{prefix}_PL_{name}": spec for name, spec in aggregations.items()}
    loan = df.groupby(["SK_ID_CURR", "SK_ID_PREV"]).agg(**named_aggs).reset_index()
    cols = [c for c in loan.columns if c.startswith(f"{prefix}_PL_")]
    pl = loan.groupby("SK_ID_CURR")[cols].agg(["mean", "max", "std"])
    pl.columns = [f"{c[0]}_{c[1].upper()}" for c in pl.columns]
    return feat.merge(pl.reset_index(), on="SK_ID_CURR", how="left")


def credit_card_features(config: TrainingConfig, raw_data_dir: Path = RAW_DATA_DIR):
    cc = read_table("credit_card_balance", config, raw_data_dir)
    cc["CC_BAL_LIM_RATIO"] = cc["AMT_BALANCE"] / (cc["AMT_CREDIT_LIMIT_ACTUAL"] + 1)
    cc["CC_PAY_TOTAL_RATIO"] = cc["AMT_PAYMENT_TOTAL_CURRENT"] / (cc["AMT_TOTAL_RECEIVABLE"] + 1)
    cc["CC_DRAW_LIM"] = cc["AMT_DRAWINGS_CURRENT"] / (cc["AMT_CREDIT_LIMIT_ACTUAL"] + 1)
    cc["CC_LATE"] = (cc["SK_DPD"] > 0).astype("int8")
    cc["CC_MIN_PAY_RATIO"] = cc["AMT_INST_MIN_REGULARITY"] / (cc["AMT_PAYMENT_CURRENT"] + 1)
    feat = _base_group_agg(cc, "CC", ["SK_ID_CURR", "SK_ID_PREV"])
    feat = feat.merge(cc.groupby("SK_ID_CURR")["CC_LATE"].mean().reset_index(name="CC_LATE_RATE"), on="SK_ID_CURR", how="left")
    for months, label in [(-3, "3M"), (-6, "6M"), (-12, "12M"), (-24, "24M")]:
        feat = feat.merge(agg_time_window(cc, "SK_ID_CURR", ["AMT_BALANCE", "CC_BAL_LIM_RATIO", "CC_DRAW_LIM", "SK_DPD"], "MONTHS_BALANCE", months, f"CC_{label}"), on="SK_ID_CURR", how="left")
    feat = _merge_per_loan(feat, cc, "CC", {"BAL_LIM_MAX": ("CC_BAL_LIM_RATIO", "max"), "BAL_LIM_MEAN": ("CC_BAL_LIM_RATIO", "mean"), "DRAW_MEAN": ("CC_DRAW_LIM", "mean"), "DPD_MAX": ("SK_DPD", "max"), "LATE_RATE": ("CC_LATE", "mean")})
    feat = feat.merge(time_weighted_agg(cc, "SK_ID_CURR", ["AMT_BALANCE", "CC_BAL_LIM_RATIO", "SK_DPD"], "MONTHS_BALANCE", "CC", decay=0.02), on="SK_ID_CURR", how="left")
    for col in ["AMT_BALANCE", "CC_BAL_LIM_RATIO"]:
        feat = feat.merge(compute_trend(cc, "SK_ID_CURR", col, "MONTHS_BALANCE", "CC"), on="SK_ID_CURR", how="left")
    return feat


def installments_features(config: TrainingConfig, raw_data_dir: Path = RAW_DATA_DIR):
    ins = read_table("installments_payments", config, raw_data_dir)
    ins["PAYMENT_PERC"] = (ins["AMT_PAYMENT"] / (ins["AMT_INSTALMENT"] + 0.001)).replace([np.inf, -np.inf], np.nan).astype("float32")
    ins["PAYMENT_DIFF"] = (ins["AMT_INSTALMENT"] - ins["AMT_PAYMENT"]).astype("float32")
    ins["DPD"] = np.maximum(ins["DAYS_ENTRY_PAYMENT"] - ins["DAYS_INSTALMENT"], 0).astype("float32")
    ins["DBD"] = np.maximum(ins["DAYS_INSTALMENT"] - ins["DAYS_ENTRY_PAYMENT"], 0).astype("float32")
    ins["LATE_PAYMENT"] = (ins["DPD"] > 0).astype("int8")
    ins["SIGNIFICANT_UNDERPAY"] = (ins["PAYMENT_DIFF"] > 100).astype("int8")
    feat = _base_group_agg(ins, "INS", ["SK_ID_CURR", "SK_ID_PREV"])
    feat = feat.merge(ins.groupby("SK_ID_CURR")["LATE_PAYMENT"].mean().reset_index(name="INS_LATE_RATE"), on="SK_ID_CURR", how="left")
    feat = feat.merge(ins.groupby("SK_ID_CURR")["SIGNIFICANT_UNDERPAY"].mean().reset_index(name="INS_SIGUNDERPAY_RATE"), on="SK_ID_CURR", how="left")
    for days, label in [(-180, "6M"), (-365, "1Y"), (-730, "2Y")]:
        feat = feat.merge(agg_time_window(ins, "SK_ID_CURR", ["DPD", "PAYMENT_PERC", "PAYMENT_DIFF", "LATE_PAYMENT"], "DAYS_INSTALMENT", days, f"INS_{label}"), on="SK_ID_CURR", how="left")
    feat = _merge_per_loan(feat, ins, "INS", {"DPD_MEAN": ("DPD", "mean"), "DPD_MAX": ("DPD", "max"), "LATE_SUM": ("LATE_PAYMENT", "sum"), "LATE_RATE": ("LATE_PAYMENT", "mean"), "PAYPERC_MEAN": ("PAYMENT_PERC", "mean"), "PAYPERC_MIN": ("PAYMENT_PERC", "min"), "PAYDIFF_MAX": ("PAYMENT_DIFF", "max"), "COUNT": ("DPD", "size")})
    feat = feat.merge(time_weighted_agg(ins, "SK_ID_CURR", ["DPD", "PAYMENT_PERC", "PAYMENT_DIFF"], "DAYS_INSTALMENT", "INS", decay=0.001), on="SK_ID_CURR", how="left")
    for col in ["DPD", "PAYMENT_PERC"]:
        feat = feat.merge(compute_trend(ins, "SK_ID_CURR", col, "DAYS_INSTALMENT", "INS"), on="SK_ID_CURR", how="left")
    ins_sorted = ins.sort_values("DAYS_INSTALMENT", ascending=False)
    for k in [3, 5, 10, 30]:
        last_k = ins_sorted.groupby("SK_ID_CURR").head(k)
        lk = last_k.groupby("SK_ID_CURR").agg(
            **{
                f"INS_LAST{k}_DPD_MEAN": ("DPD", "mean"),
                f"INS_LAST{k}_DPD_MAX": ("DPD", "max"),
                f"INS_LAST{k}_PAYPERC_MEAN": ("PAYMENT_PERC", "mean"),
                f"INS_LAST{k}_PAYDIFF_MEAN": ("PAYMENT_DIFF", "mean"),
                f"INS_LAST{k}_LATE_RATE": ("LATE_PAYMENT", "mean"),
            }
        ).reset_index()
        feat = feat.merge(lk, on="SK_ID_CURR", how="left")
    if "NUM_INSTALMENT_VERSION" in ins.columns:
        feat = feat.merge(ins.groupby("SK_ID_CURR")["NUM_INSTALMENT_VERSION"].agg(INS_VERSION_NUNIQUE="nunique", INS_VERSION_MAX="max", INS_VERSION_MEAN="mean").reset_index(), on="SK_ID_CURR", how="left")
    return feat


def sub_model_features(table: str, target_df, prefix, config: TrainingConfig, raw_data_dir: Path = RAW_DATA_DIR):
    df = read_csv_optimized(table_path(table, raw_data_dir), nrows=config.row_limit(f"sub_model_{table}"))
    for col in df.columns:
        if "DAYS_" in col:
            df[col] = df[col].replace(365243, np.nan)
    feat_cols = [c for c in df.columns if df[c].dtype != "object" and c not in ["SK_ID_CURR", "SK_ID_PREV", "SK_ID_BUREAU"]]
    mask_train = df["SK_ID_CURR"].isin(set(target_df["SK_ID_CURR"].values))
    df_train = df[mask_train].merge(target_df[["SK_ID_CURR", "TARGET"]], on="SK_ID_CURR", how="inner")
    df_test = df[~mask_train].copy()
    X_train = df_train[feat_cols].replace([np.inf, -np.inf], np.nan)
    y_train = df_train["TARGET"].astype(int)
    groups = df_train["SK_ID_CURR"]
    oof = np.zeros(len(df_train))
    models = []
    for tr_idx, va_idx in GroupKFold(n_splits=config.n_folds).split(X_train, y_train, groups=groups):
        m = lgb.LGBMClassifier(n_estimators=2000, learning_rate=0.05, num_leaves=31, max_depth=5, subsample=0.8, colsample_bytree=0.5, reg_alpha=0.1, reg_lambda=0.1, min_child_samples=100, random_state=config.seed, n_jobs=-1, verbose=-1)
        m.fit(X_train.iloc[tr_idx], y_train.iloc[tr_idx], eval_set=[(X_train.iloc[va_idx], y_train.iloc[va_idx])], eval_metric="auc", callbacks=[lgb.early_stopping(50, verbose=False)])
        oof[va_idx] = m.predict_proba(X_train.iloc[va_idx])[:, 1]
        models.append(m)
    print(f"  {prefix} sub-model OOF AUC: {roc_auc_score(y_train, oof):.4f}")
    df_train["_SUB_PRED"] = oof
    X_test = df_test[feat_cols].replace([np.inf, -np.inf], np.nan)
    preds = np.zeros(len(df_test))
    for model in models:
        preds += model.predict_proba(X_test)[:, 1] / len(models)
    df_test["_SUB_PRED"] = preds
    all_rows = pd.concat([df_train[["SK_ID_CURR", "_SUB_PRED"]], df_test[["SK_ID_CURR", "_SUB_PRED"]]], axis=0)
    out = all_rows.groupby("SK_ID_CURR")["_SUB_PRED"].agg(**{f"{prefix}_SUB_MEAN": "mean", f"{prefix}_SUB_MAX": "max", f"{prefix}_SUB_MIN": "min", f"{prefix}_SUB_STD": "std"}).reset_index()
    high_risk = all_rows[all_rows["_SUB_PRED"] > 0.15].groupby("SK_ID_CURR").size().reset_index(name=f"{prefix}_SUB_HIGHRISK")
    out = out.merge(high_risk, on="SK_ID_CURR", how="left")
    out[f"{prefix}_SUB_HIGHRISK"] = out[f"{prefix}_SUB_HIGHRISK"].fillna(0)
    return out


def build_feature_frame(config: TrainingConfig, raw_data_dir: Path = RAW_DATA_DIR):
    app_train = application_features(read_table("application_train", config, raw_data_dir))
    app_test = application_features(read_table("application_test", config, raw_data_dir))
    target_df = app_train[["SK_ID_CURR", "TARGET"]].copy()
    feature_tables = [
        bureau_and_balance_features(config, raw_data_dir),
        previous_application_features(config, raw_data_dir),
        pos_cash_features(config, raw_data_dir),
        credit_card_features(config, raw_data_dir),
        installments_features(config, raw_data_dir),
        sub_model_features("previous_application", target_df, "PREV", config, raw_data_dir),
        sub_model_features("bureau", target_df, "BURO", config, raw_data_dir),
        sub_model_features("installments_payments", target_df, "INS", config, raw_data_dir),
    ]
    train = app_train.copy()
    test = app_test.copy()
    for feat in feature_tables:
        train = train.merge(feat, on="SK_ID_CURR", how="left")
        test = test.merge(feat, on="SK_ID_CURR", how="left")
        del feat
        gc.collect()
    return add_global_features(train, test, config)


def add_global_features(train, test, config: TrainingConfig):
    groupby_cat_cols = ["NAME_EDUCATION_TYPE", "ORGANIZATION_TYPE", "OCCUPATION_TYPE", "NAME_INCOME_TYPE", "CODE_GENDER", "AGE_RANGE"]
    groupby_num_cols = ["AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", "EXT_MEAN", "CREDIT_ANNUITY_RATIO", "ANNUITY_INCOME_RATIO", "DAYS_EMPLOYED_YRS"]
    train, test = add_frequency_features(train, test, [c for c in train.columns if train[c].dtype == "object"])
    for c1, c2 in [("NAME_EDUCATION_TYPE", "NAME_INCOME_TYPE"), ("CODE_GENDER", "NAME_FAMILY_STATUS"), ("OCCUPATION_TYPE", "ORGANIZATION_TYPE"), ("AGE_RANGE", "NAME_EDUCATION_TYPE")]:
        if c1 in train.columns and c2 in train.columns:
            new_col = f"{c1}__{c2}"
            train[new_col] = train[c1].astype(str) + "__" + train[c2].astype(str)
            test[new_col] = test[c1].astype(str) + "__" + test[c2].astype(str)
    train, test = add_groupby_ratio_features(train, test, [c for c in groupby_cat_cols if c in train.columns], [c for c in groupby_num_cols if c in train.columns])
    te_cols = [
        c
        for c in [
            "NAME_EDUCATION_TYPE",
            "ORGANIZATION_TYPE",
            "OCCUPATION_TYPE",
            "NAME_INCOME_TYPE",
            "CODE_GENDER",
            "NAME_HOUSING_TYPE",
            "AGE_RANGE",
            "NAME_EDUCATION_TYPE__NAME_INCOME_TYPE",
            "CODE_GENDER__NAME_FAMILY_STATUS",
            "OCCUPATION_TYPE__ORGANIZATION_TYPE",
            "AGE_RANGE__NAME_EDUCATION_TYPE",
            "NAME_FAMILY_STATUS",
            "NAME_CONTRACT_TYPE",
        ]
        if c in train.columns
    ]
    train, test = add_target_encoding(train, test, "TARGET", te_cols, n_splits=config.n_folds, smoothing=config.te_smoothing, min_samples_leaf=config.te_min_samples, seed=config.seed)
    train = reduce_memory_usage(train)
    test = reduce_memory_usage(test)
    return add_knn_features(train, test, config)


def add_knn_features(train, test, config: TrainingConfig):
    target = train["TARGET"].astype("int8")
    knn_cols = ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3", "CREDIT_ANNUITY_RATIO"]
    x_all = pd.concat([train[knn_cols], test[knn_cols]], axis=0).fillna(-999).values
    x_all = StandardScaler().fit_transform(x_all)
    x_tr = x_all[: len(train)]
    x_te = x_all[len(train) :]
    skf = StratifiedKFold(n_splits=config.n_folds, shuffle=True, random_state=config.seed)
    for k in config.knn_neighbors:
        oof = np.zeros(len(train))
        pred = np.zeros(len(test))
        for tr_idx, va_idx in skf.split(x_tr, target):
            knn = KNeighborsClassifier(n_neighbors=k, metric="euclidean", n_jobs=-1)
            knn.fit(x_tr[tr_idx], target.values[tr_idx])
            oof[va_idx] = knn.predict_proba(x_tr[va_idx])[:, 1]
            pred += knn.predict_proba(x_te)[:, 1] / config.n_folds
        train[f"KNN_TARGET_{k}"] = oof.astype("float32")
        test[f"KNN_TARGET_{k}"] = pred.astype("float32")
        print(f"  KNN k={k} OOF AUC: {roc_auc_score(target, oof):.6f}")
    return train, test
