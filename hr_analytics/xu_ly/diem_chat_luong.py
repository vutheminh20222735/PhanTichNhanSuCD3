"""Đánh giá chất lượng dữ liệu động."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from xu_ly.lam_sach import detect_outliers_iqr


def calculate_quality_score(df: pd.DataFrame, id_col: str | None = None) -> dict[str, Any]:
    """Tính Completeness / Consistency / Validity / Uniqueness từ dữ liệu thật."""
    if df is None or df.empty:
        return {
            "score": 0.0,
            "label": "No data",
            "completeness": 0.0,
            "consistency": 0.0,
            "validity": 0.0,
            "uniqueness": 0.0,
            "issues": ["Dataset rỗng."],
            "message": "No dataset loaded.",
        }

    n_rows, n_cols = df.shape
    total_cells = max(n_rows * n_cols, 1)
    missing = int(df.isna().sum().sum())
    empty_str = 0
    for col in df.select_dtypes(include=["object", "string"]).columns:
        empty_str += int((df[col].astype(str).str.strip() == "").sum())

    completeness = max(0.0, 100.0 * (1 - (missing + empty_str) / total_cells))

    dup = int(df.duplicated().sum())
    uniqueness = max(0.0, 100.0 * (1 - dup / max(n_rows, 1)))
    if id_col and id_col in df.columns:
        id_dup = int(df[id_col].duplicated().sum())
        uniqueness = min(uniqueness, max(0.0, 100.0 * (1 - id_dup / max(n_rows, 1))))

    # Validity: Inf + numeric coerce failures on object-looking numerics
    inf_count = 0
    invalid_numeric = 0
    for col in df.columns:
        s = df[col]
        if pd.api.types.is_numeric_dtype(s):
            inf_count += int(np.isinf(pd.to_numeric(s, errors="coerce")).sum())
        else:
            coerced = pd.to_numeric(s, errors="coerce")
            # only count if mostly numeric-looking
            if s.notna().any() and coerced.notna().mean() >= 0.5:
                invalid_numeric += int(s.notna().sum() - coerced.notna().sum())

    validity_penalty = (inf_count + invalid_numeric) / max(n_rows, 1)
    validity = max(0.0, 100.0 * (1 - min(validity_penalty, 1.0)))

    # Consistency: constant columns + mixed-type hints
    const_cols = [c for c in df.columns if df[c].nunique(dropna=True) <= 1]
    consistency = max(0.0, 100.0 - min(40.0, len(const_cols) * 8.0))

    score = round(0.35 * completeness + 0.25 * uniqueness + 0.25 * validity + 0.15 * consistency, 1)
    if score >= 90:
        label = "Good"
        message = "Dataset quality is good."
    elif score >= 70:
        label = "Fair"
        message = "Dataset has some quality issues."
    else:
        label = "Needs attention"
        message = "Dataset has significant quality issues."

    issues: list[str] = []
    if missing:
        issues.append(f"Missing cells: {missing:,}")
    if empty_str:
        issues.append(f"Empty strings: {empty_str:,}")
    if dup:
        issues.append(f"Duplicate rows: {dup:,}")
    if id_col and id_col in df.columns:
        id_dup = int(df[id_col].duplicated().sum())
        if id_dup:
            issues.append(f"Duplicate IDs in `{id_col}`: {id_dup:,}")
    if inf_count:
        issues.append(f"Inf/-Inf values: {inf_count:,}")
    if invalid_numeric:
        issues.append(f"Invalid numeric-like values: {invalid_numeric:,}")
    if const_cols:
        issues.append(f"Constant columns: {', '.join(const_cols[:8])}")

    outliers = detect_outliers_iqr(df, columns=list(df.select_dtypes(include=[np.number]).columns)[:20])
    out_n = int(outliers["Outlier Count"].sum()) if not outliers.empty else 0
    if out_n:
        issues.append(f"IQR outliers (reported, not removed): {out_n:,}")

    return {
        "score": score,
        "label": label,
        "message": message,
        "completeness": round(completeness, 1),
        "consistency": round(consistency, 1),
        "validity": round(validity, 1),
        "uniqueness": round(uniqueness, 1),
        "issues": issues,
        "missing": missing,
        "duplicates": dup,
        "empty_strings": empty_str,
        "outliers": outliers,
    }
