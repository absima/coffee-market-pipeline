from __future__ import annotations

import argparse
import os

import pandas as pd
from sqlalchemy import create_engine


def main(in_csv: str, db_path: str, table_name: str) -> None:
    """
    Read a (date, value) CSV file and write it to a SQLite table.

    Parameters
    ----------
    in_csv:
        Path to the input CSV file.
    db_path:
        Path to the SQLite database file.
    table_name:
        Name of the table to create or replace.
    """
    os.makedirs(os.path.dirname(db_path), exist_ok=True)

    df = pd.read_csv(in_csv, parse_dates=["date"])
    if not {"date", "value"}.issubset(df.columns):
        raise ValueError(
            f"Expected columns date,value in {in_csv}. Got: {df.columns.tolist()}"
        )

    engine = create_engine(f"sqlite:///{db_path}")
    df.to_sql(table_name, engine, if_exists="replace", index=False)

    print(f"Loaded {len(df):,} rows into sqlite:///{db_path} table={table_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-csv", required=True)
    parser.add_argument("--db-path", required=True)
    parser.add_argument("--table", required=True)
    args = parser.parse_args()

    main(args.in_csv, args.db_path, args.table)
