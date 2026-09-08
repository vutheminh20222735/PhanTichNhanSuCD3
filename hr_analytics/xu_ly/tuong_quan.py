"""Phân tích tương quan — dùng mọi cột numeric của dataframe hiện tại."""

from __future__ import annotations

from typing import Any

import pandas as pd

from xu_ly.tim_cot_du_lieu import map_binary_target


def get_numeric_frame(
    df: pd.DataFrame,
    include_target: bool = True,
    target: str | None = None,
) -> pd.DataFrame:
    cols = list(df.select_dtypes(include="number").columns)
    out = df[cols].apply(pd.to_numeric, errors="coerce") if cols else pd.DataFrame(index=df.index)
    if include_target and target and target in df.columns and target not in out.columns:
        try:
            out = out.copy()
            out[f"{target}_Encoded"] = map_binary_target(df[target])
        except Exception:  # noqa: BLE001
            pass
    return out


def compute_correlation_matrix(
    df: pd.DataFrame,
    include_target: bool = True,
    target: str | None = None,
) -> pd.DataFrame:
    frame = get_numeric_frame(df, include_target=include_target, target=target)
    if frame.empty or frame.shape[1] < 2:
        return pd.DataFrame()
    return frame.corr(method="pearson")


def find_strong_correlations(corr: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
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
    strong = find_strong_correlations(corr, threshold=threshold)
    if strong.empty:
        return [f"Không có cặp biến số nào có |correlation| ≥ {threshold}."]
    insights = []
    for _, row in strong.head(10).iterrows():
        direction = "thuận" if row["Correlation"] > 0 else "nghịch"
        insights.append(
            f"`{row['Biến A']}` liên hệ {direction} với `{row['Biến B']}` "
            f"(r = {row['Correlation']}) — tương quan quan sát, không khẳng định nhân quả."
        )
    return insights


def analyze_correlation(df: pd.DataFrame, target: str | None = None) -> dict[str, Any]:
    corr = compute_correlation_matrix(df, target=target)
    return {
        "matrix": corr,
        "strong_pairs": find_strong_correlations(corr, threshold=0.5) if not corr.empty else pd.DataFrame(),
        "insights": correlation_insights(corr) if not corr.empty else ["Không đủ biến số để tính correlation."],
    }
