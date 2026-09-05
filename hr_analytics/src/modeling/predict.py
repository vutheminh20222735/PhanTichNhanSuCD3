"""Dự báo nguy cơ nghỉ việc cho nhân viên."""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.utils import LEAVE_LABEL, STAY_LABEL


def risk_band(probability: float) -> str:
    """0–30% Low, 30–60% Medium, ≥60% High."""
    pct = probability * 100
    if pct < 30:
        return "Low"
    if pct < 60:
        return "Medium"
    return "High"


def predict_attrition(
    model,
    employee_features: dict[str, Any] | pd.DataFrame,
    feature_columns: list[str] | None = None,
) -> dict[str, Any]:
    if isinstance(employee_features, dict):
        row = pd.DataFrame([employee_features])
    else:
        row = employee_features.copy()

    if feature_columns is not None:
        missing = [c for c in feature_columns if c not in row.columns]
        if missing:
            raise ValueError(f"Thiếu feature: {missing}")
        row = row[feature_columns]

    if hasattr(model, "predict_proba"):
        proba = float(model.predict_proba(row)[0, 1])
    else:
        proba = float(model.predict(row)[0])

    pred_label = LEAVE_LABEL if proba >= 0.5 else STAY_LABEL
    band = risk_band(proba)

    return {
        "probability": proba,
        "probability_pct": round(proba * 100, 1),
        "prediction": pred_label,
        "risk_band": band,
        "message": (
            f"Nguy cơ nghỉ việc: {proba * 100:.1f}%\n"
            f"Dự đoán: {pred_label}\n"
            f"Mức rủi ro: {band}"
        ),
    }
