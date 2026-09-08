"""Thống kê mô tả động theo cột có trong dataframe."""

from __future__ import annotations

import pandas as pd


def descriptive_statistics(
    df: pd.DataFrame,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    cols = columns or list(df.select_dtypes(include="number").columns)
    cols = [c for c in cols if c in df.columns]
    if not cols:
        return pd.DataFrame()
    desc = df[cols].apply(pd.to_numeric, errors="coerce").describe().T
    desc = desc.reset_index().rename(columns={"index": "Column"})
    return desc
