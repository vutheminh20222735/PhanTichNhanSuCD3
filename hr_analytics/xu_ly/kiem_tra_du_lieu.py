"""Kiểm tra chất lượng dữ liệu — áp dụng cho mọi dataset."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def check_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    missing = df.isnull().sum()
    pct = (missing / len(df) * 100) if len(df) else missing
    result = pd.DataFrame(
        {
            "Column": missing.index,
            "Missing Count": missing.values,
            "Missing %": [round(float(x), 2) for x in pct.values],
        }
    )
    return result.sort_values("Missing Count", ascending=False).reset_index(drop=True)


def check_duplicates(df: pd.DataFrame) -> dict[str, Any]:
    n_dup = int(df.duplicated().sum())
    return {
        "duplicate_count": n_dup,
        "duplicate_pct": round(n_dup / len(df) * 100, 2) if len(df) else 0.0,
        "message": (
            "Không phát hiện duplicate records."
            if n_dup == 0
            else f"Phát hiện {n_dup} bản ghi trùng lặp."
        ),
    }


def check_data_types(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Column": df.columns,
            "Data Type": [str(df[c].dtype) for c in df.columns],
        }
    )


def get_categorical_uniques(df: pd.DataFrame, max_unique: int = 50) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]) and df[col].nunique(dropna=True) > 20:
            continue
        nunique = int(df[col].nunique(dropna=True))
        if nunique == 0 or nunique > max_unique:
            continue
        uniques = sorted(df[col].dropna().astype(str).unique().tolist())
        result[col] = {"n_categories": len(uniques), "categories": uniques}
    return result


def check_invalid_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Kiểm tra Inf và giá trị không parse được trên cột số."""
    rows = []
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            series = pd.to_numeric(df[col], errors="coerce")
            n_inf = int(np.isinf(series.to_numpy(dtype=float, na_value=np.nan)).sum())
            n_neg = int((series < 0).sum()) if series.notna().any() else 0
            rows.append(
                {
                    "Column": col,
                    "Inf Count": n_inf,
                    "Negative Count": n_neg,
                    "Invalid Count": n_inf,
                    "Invalid %": round(n_inf / len(df) * 100, 2) if len(df) else 0.0,
                }
            )
        else:
            coerced = pd.to_numeric(df[col], errors="coerce")
            if df[col].notna().any() and coerced.notna().mean() >= 0.5:
                n_bad = int(df[col].notna().sum() - coerced.notna().sum())
                rows.append(
                    {
                        "Column": col,
                        "Inf Count": 0,
                        "Negative Count": 0,
                        "Invalid Count": n_bad,
                        "Invalid %": round(n_bad / len(df) * 100, 2) if len(df) else 0.0,
                    }
                )
    return pd.DataFrame(rows)


def validate_dataset(df: pd.DataFrame) -> dict[str, Any]:
    missing_df = check_missing_values(df)
    total_missing = int(missing_df["Missing Count"].sum())
    dup = check_duplicates(df)
    invalid = check_invalid_numeric(df)
    return {
        "missing_summary": missing_df,
        "total_missing": total_missing,
        "missing_message": (
            "Không phát hiện missing values."
            if total_missing == 0
            else f"Phát hiện tổng {total_missing} giá trị thiếu."
        ),
        "duplicates": dup,
        "dtypes": check_data_types(df),
        "categorical_uniques": get_categorical_uniques(df),
        "invalid_numeric": invalid,
        "numeric_columns_present": list(df.select_dtypes(include="number").columns),
    }
