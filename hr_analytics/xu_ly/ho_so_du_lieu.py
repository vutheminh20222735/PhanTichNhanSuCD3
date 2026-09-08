"""Profile dataset động — không hard-code tên cột / số liệu."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from xu_ly.tim_cot_du_lieu import build_schema, detect_target_candidates, find_role


def _is_datetime_series(s: pd.Series) -> bool:
    if pd.api.types.is_datetime64_any_dtype(s):
        return True
    if not (pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s)):
        return False
    sample = s.dropna().astype(str).head(40)
    if sample.empty or len(sample) < 3:
        return False
    # Chỉ thử parse khi trông giống ngày (tránh warning / false positive)
    look_like = sample.str.contains(r"\d{4}[-/]\d{1,2}[-/]\d{1,2}|\d{1,2}[-/]\d{1,2}[-/]\d{2,4}", regex=True)
    if float(look_like.mean()) < 0.6:
        return False
    parsed = pd.to_datetime(sample, errors="coerce")
    return float(parsed.notna().mean()) >= 0.8


def detect_column_types(df: pd.DataFrame) -> dict[str, list[str]]:
    numeric, categorical, boolean, datetime_cols = [], [], [], []
    for col in df.columns:
        s = df[col]
        if pd.api.types.is_bool_dtype(s):
            boolean.append(col)
        elif _is_datetime_series(s):
            datetime_cols.append(col)
        elif pd.api.types.is_numeric_dtype(s):
            numeric.append(col)
        else:
            # numeric stored as string?
            coerced = pd.to_numeric(s, errors="coerce")
            if s.notna().any() and coerced.notna().mean() >= 0.9:
                numeric.append(col)
            else:
                categorical.append(col)
    return {
        "numeric": numeric,
        "categorical": categorical,
        "boolean": boolean,
        "datetime": datetime_cols,
    }


def profile_dataset(df: pd.DataFrame, target: str | None = None) -> dict[str, Any]:
    """Tạo profile đầy đủ từ dataframe hiện tại."""
    if df is None or df.empty:
        return {
            "row_count": 0,
            "column_count": 0,
            "numeric_count": 0,
            "categorical_count": 0,
            "datetime_count": 0,
            "boolean_count": 0,
            "missing_cells": 0,
            "duplicate_rows": 0,
            "columns": [],
            "constant_columns": [],
            "high_cardinality_columns": [],
            "id_columns": [],
            "target_candidates": [],
            "types": {"numeric": [], "categorical": [], "boolean": [], "datetime": []},
        }

    types = detect_column_types(df)
    n_rows, n_cols = df.shape
    missing_cells = int(df.isna().sum().sum())
    dup_rows = int(df.duplicated().sum())

    columns_meta = []
    constant_columns = []
    high_card = []
    id_like = []

    for col in df.columns:
        s = df[col]
        nunique = int(s.nunique(dropna=True))
        miss = int(s.isna().sum())
        dtype = str(s.dtype)
        if col in types["datetime"]:
            kind = "datetime"
        elif col in types["boolean"]:
            kind = "boolean"
        elif col in types["numeric"]:
            kind = "numeric"
        else:
            kind = "categorical"

        columns_meta.append(
            {
                "Column": col,
                "Type": kind,
                "Dtype": dtype,
                "Non-null": int(s.notna().sum()),
                "Missing": miss,
                "Missing %": round(miss / n_rows * 100, 2) if n_rows else 0.0,
                "Unique": nunique,
                "Cardinality": nunique,
            }
        )
        if nunique <= 1:
            constant_columns.append(col)
        if kind == "categorical" and n_rows and nunique / n_rows >= 0.9:
            high_card.append(col)
        col_l = col.lower().replace(" ", "")
        if any(k in col_l for k in ("id", "employee", "manhanvien", "empid")) and nunique >= max(n_rows * 0.9, 1):
            id_like.append(col)

    id_col = find_role(df, "id")
    if id_col and id_col not in id_like:
        id_like.insert(0, id_col)

    candidates = detect_target_candidates(df)
    schema = build_schema(df, target=target)

    return {
        "row_count": n_rows,
        "column_count": n_cols,
        "numeric_count": len(types["numeric"]),
        "categorical_count": len(types["categorical"]),
        "datetime_count": len(types["datetime"]),
        "boolean_count": len(types["boolean"]),
        "missing_cells": missing_cells,
        "duplicate_rows": dup_rows,
        "columns": columns_meta,
        "constant_columns": constant_columns,
        "high_cardinality_columns": high_card,
        "id_columns": id_like,
        "target_candidates": candidates,
        "types": types,
        "schema": schema,
        "memory_mb": float(df.memory_usage(deep=True).sum() / (1024**2)),
    }


def suggest_filter_columns(df: pd.DataFrame, schema: dict[str, Any], max_n: int = 6) -> list[str]:
    """Chọn cột filter động từ roles hoặc categorical cardinality hợp lý."""
    roles = schema.get("roles") or {}
    preferred_roles = ["department", "location", "gender", "overtime", "job_role", "contract", "education"]
    cols: list[str] = []
    for role in preferred_roles:
        c = roles.get(role)
        if c and c in df.columns and c not in cols:
            cols.append(c)
    if len(cols) >= 2:
        return cols[:max_n]

    for col in df.columns:
        if col in cols:
            continue
        nunique = int(df[col].nunique(dropna=True))
        if 2 <= nunique <= 20:
            cols.append(col)
        if len(cols) >= max_n:
            break
    return cols[:max_n]


def format_file_size(n_bytes: int | None) -> str:
    if n_bytes is None:
        return "—"
    units = ["B", "KB", "MB", "GB"]
    size = float(n_bytes)
    for u in units:
        if size < 1024 or u == units[-1]:
            return f"{size:.1f} {u}" if u != "B" else f"{int(size)} {u}"
        size /= 1024
    return f"{n_bytes} B"
