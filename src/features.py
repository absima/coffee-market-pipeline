from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
from sqlalchemy import create_engine


def buildFeatureMatrix(df: pd.DataFrame, lags: list[int], windows: list[int]) -> pd.DataFrame:
    """
    Build a supervised learning dataset from a time series.

    The target is next-step log return:
        log_return[t] = log(price[t]) - log(price[t-1])
        y_next_return[t] = log_return[t+1]

    Features:
    - Lagged log returns: r_lag_k
    - Rolling mean/std of log returns: r_roll_mean_w, r_roll_std_w

    Parameters
    ----------
    df:
        DataFrame with at least columns ["date", "value"].
    lags:
        Return lags to include as features (e.g., [1, 3, 6, 12]).
    windows:
        Rolling windows to compute return statistics (e.g., [3, 6, 12]).

    Returns
    -------
    pd.DataFrame
        DataFrame containing original columns plus engineered features and target.
        Rows with missing values introduced by lag/rolling/shift are dropped.
    """
    df = df.sort_values("date").copy()

    # Ensure strictly positive values before log transform
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["value"])
    df = df[df["value"] > 0].copy()

    df["log_price"] = np.log(df["value"].astype(float))
    df["log_return"] = df["log_price"].diff()

    for lag in lags:
        df[f"r_lag_{lag}"] = df["log_return"].shift(lag)

    for w in windows:
        df[f"r_roll_mean_{w}"] = df["log_return"].rolling(window=w).mean()
        df[f"r_roll_std_{w}"] = df["log_return"].rolling(window=w).std()

    df["y_next_return"] = df["log_return"].shift(-1)

    df = df.dropna().reset_index(drop=True)
    return df


def main(db_path: str, in_table: str, out_csv: str, lags: list[int], windows: list[int]) -> None:
    """
    Read a time series from SQLite, build features/target, and write a model-ready CSV.

    Parameters
    ----------
    db_path:
        Path to the SQLite database.
    in_table:
        Name of the table to read (must contain date, value).
    out_csv:
        Output CSV path for the engineered dataset.
    lags:
        List of lag steps used for lagged-return features.
    windows:
        List of rolling window sizes used for rolling statistics features.
    """
    engine = create_engine(f"sqlite:///{db_path}")
    df = pd.read_sql_table(in_table, engine)

    required = {"date", "value"}
    if not required.issubset(df.columns):
        raise ValueError(f"Table {in_table} must contain {required}. Got: {df.columns.tolist()}")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).copy()

    feat = buildFeatureMatrix(df, lags=lags, windows=windows)

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    feat.to_csv(out_csv, index=False)
    print(f"Wrote features: {len(feat):,} rows -> {out_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--db-path", required=True)
    parser.add_argument("--in-table", required=True)
    parser.add_argument("--out-csv", required=True)
    parser.add_argument("--lags", required=True, help="Comma-separated list, e.g. 1,3,6,12")
    parser.add_argument("--windows", required=True, help="Comma-separated list, e.g. 3,6,12")
    args = parser.parse_args()

    lags = [int(x.strip()) for x in args.lags.split(",") if x.strip()]
    windows = [int(x.strip()) for x in args.windows.split(",") if x.strip()]
    main(args.db_path, args.in_table, args.out_csv, lags, windows)
