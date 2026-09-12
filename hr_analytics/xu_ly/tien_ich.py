"""Tiện ích dùng chung — không gắn với dataset cụ thể."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models" / "saved_models"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RAW_DIR = PROJECT_ROOT / "data" / "raw"
DEFAULT_DATASET_PATH = RAW_DIR / "du_lieu_nhan_su_tieng_viet_day_du.csv"

LEAVE_LABEL = "Có"
STAY_LABEL = "Không"


def format_vnd(value: float | int | None) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"{int(round(float(value))):,} VND".replace(",", ".")


def format_vnd_short(value: float | int | None) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"{float(value) / 1_000_000:.1f} triệu VND"


def format_vnd_compact(value: float | int | None) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"{float(value) / 1_000_000:.2f}M ₫"


def format_pct(value: float | None, digits: int = 2) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"{float(value):.{digits}f}%"


def attrition_mask(series: pd.Series) -> pd.Series:
    """Mask lớp 'leave' linh hoạt theo giá trị phổ biến."""
    from xu_ly.tim_cot_du_lieu import POSITIVE_LEAVE_VALUES, map_binary_target

    try:
        return map_binary_target(series).astype(bool)
    except Exception:  # noqa: BLE001
        s = series.astype(str).str.strip().str.lower()
        return s.isin(POSITIVE_LEAVE_VALUES)


def attrition_rate(df: pd.DataFrame, target: str) -> float:
    if df.empty or not target or target not in df.columns:
        return 0.0
    return float(attrition_mask(df[target]).mean() * 100)


def safe_div(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def apply_filters(df: pd.DataFrame, filters: dict[str, Any]) -> pd.DataFrame:
    out = df
    for col, selected in filters.items():
        if col not in out.columns or selected is None:
            continue
        if isinstance(selected, (list, tuple, set)):
            if len(selected) == 0:
                continue
            want = {str(x) for x in selected}
            out = out[out[col].astype(str).isin(want)]
        else:
            out = out[out[col].astype(str) == str(selected)]
    if out is df:
        return df.copy()
    return out.reset_index(drop=True)
