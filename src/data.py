from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.config import RAW_DATA_DIR, TABLE_FILES, TrainingConfig


def read_csv_optimized(path: str | Path, usecols: list[str] | None = None, nrows: int | None = None):
    df = pd.read_csv(path, usecols=usecols, nrows=nrows)
    for col in df.columns:
        if df[col].dtype == "float64":
            df[col] = df[col].astype("float32")
        elif df[col].dtype == "int64":
            if df[col].min() >= np.iinfo(np.int32).min and df[col].max() <= np.iinfo(np.int32).max:
                df[col] = df[col].astype("int32")
    return df


def reduce_memory_usage(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        col_type = df[col].dtype
        if str(col_type).startswith("float"):
            df[col] = pd.to_numeric(df[col], downcast="float")
        elif str(col_type).startswith("int"):
            df[col] = pd.to_numeric(df[col], downcast="integer")
    return df


def table_path(table_name: str, raw_data_dir: Path = RAW_DATA_DIR) -> Path:
    return raw_data_dir / TABLE_FILES[table_name]


def read_table(
    table_name: str,
    config: TrainingConfig,
    raw_data_dir: Path = RAW_DATA_DIR,
    usecols: list[str] | None = None,
) -> pd.DataFrame:
    return read_csv_optimized(
        table_path(table_name, raw_data_dir),
        usecols=usecols,
        nrows=config.row_limit(table_name),
    )
