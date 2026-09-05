"""Phân tích tương quan giữa các biến số."""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.utils import NUMERIC_COLUMNS, TARGET_COL
from src.preprocessing.transformer import map_target


def get_numeric_frame(df: pd.DataFrame, include_target: bool = True) -> pd.DataFrame:
    """Lấy khung dữ liệu số cho correlation."""
    cols = [c for c in NUMERIC_COLUMNS if c in df.columns]
    out = df[cols].apply(pd.to_numeric, errors="coerce")
    if include_target and TARGET_COL in df.columns:
        out = out.copy()
        out["NghiViec_Encoded"] = map_target(df[TARGET_COL])
    return out


def compute_correlation_matrix(df: pd.DataFrame, include_target: bool = True) -> pd.DataFrame:
    """Tính correlation matrix (Pearson) trên biến số."""
    frame = get_numeric_frame(df, include_target=include_target)
    return frame.corr(method="pearson")


def find_strong_correlations(
    corr: pd.DataFrame,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """Tìm các cặp biến có |corr| >= threshold (không gồm đường chéo)."""
    pairs = []
    cols = list(corr.columns)
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            val = corr.iloc[i, j]
            if pd.isna(val):
                continue
            if abs(val) >= threshold:
                pairs.append(
                    {
                        "Biến A": cols[i],
                        "Biến B": cols[j],
                        "Correlation": round(float(val), 3),
                        "|Correlation|": round(abs(float(val)), 3),
                    }
                )
    result = pd.DataFrame(pairs)
    if result.empty:
        return result
    return result.sort_values("|Correlation|", ascending=False).reset_index(drop=True)


def correlation_insights(corr: pd.DataFrame, threshold: float = 0.5) -> list[str]:
    """Sinh nhận xét correlation — tránh ngôn ngữ nhân quả."""
    strong = find_strong_correlations(corr, threshold=threshold)
    insights: list[str] = []
    if strong.empty:
        insights.append(
            f"Không có cặp biến số nào có |correlation| ≥ {threshold} trong dataset hiện tại."
        )
        return insights

    for _, row in strong.head(10).iterrows():
        direction = "thuận" if row["Correlation"] > 0 else "nghịch"
        insights.append(
            f"`{row['Biến A']}` có mối liên hệ {direction} với `{row['Biến B']}` "
            f"(r = {row['Correlation']}). Đây là tương quan quan sát được, "
            "không khẳng định quan hệ nhân quả."
        )
    return insights


def analyze_correlation(df: pd.DataFrame) -> dict[str, Any]:
    """Tổng hợp phân tích correlation."""
    corr = compute_correlation_matrix(df)
    strong = find_strong_correlations(corr, threshold=0.5)
    return {
        "matrix": corr,
        "strong_pairs": strong,
        "insights": correlation_insights(corr),
    }
