"""Thống kê mô tả."""

from __future__ import annotations

import pandas as pd

from src.utils import DESCRIPTIVE_COLUMNS, TARGET_COL, attrition_mask, format_vnd


def descriptive_statistics(
    df: pd.DataFrame,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """Tính count, mean, median, std, min, max, Q1, Q3."""
    cols = columns or DESCRIPTIVE_COLUMNS
    rows = []
    for col in cols:
        if col not in df.columns:
            continue
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        if s.empty:
            continue
        rows.append(
            {
                "Biến": col,
                "count": int(s.count()),
                "mean": round(float(s.mean()), 2),
                "median": round(float(s.median()), 2),
                "std": round(float(s.std()), 2),
                "min": round(float(s.min()), 2),
                "Q1": round(float(s.quantile(0.25)), 2),
                "Q3": round(float(s.quantile(0.75)), 2),
                "max": round(float(s.max()), 2),
            }
        )
    return pd.DataFrame(rows)


def compute_kpis(df: pd.DataFrame) -> dict[str, float | int | str]:
    """Tính KPI dashboard từ dataframe (có thể đã filter)."""
    total = len(df)
    if total == 0 or TARGET_COL not in df.columns:
        return {
            "total_employees": 0,
            "employees_staying": 0,
            "employees_leaving": 0,
            "attrition_rate": 0.0,
            "avg_age": 0.0,
            "avg_income": 0.0,
            "avg_income_fmt": "N/A",
            "avg_years_company": 0.0,
            "avg_job_satisfaction": 0.0,
        }

    leaving = int(attrition_mask(df[TARGET_COL]).sum())
    staying = total - leaving
    attrition = leaving / total * 100

    def _mean(col: str) -> float:
        if col not in df.columns:
            return 0.0
        return float(pd.to_numeric(df[col], errors="coerce").mean())

    avg_income = _mean("ThuNhapHangThang_VND")
    return {
        "total_employees": total,
        "employees_staying": staying,
        "employees_leaving": leaving,
        "attrition_rate": round(attrition, 2),
        "avg_age": round(_mean("Tuoi"), 2),
        "avg_income": avg_income,
        "avg_income_fmt": format_vnd(avg_income),
        "avg_years_company": round(_mean("SoNamTaiCongTy"), 2),
        "avg_job_satisfaction": round(_mean("HaiLong_CongViec"), 2),
    }
