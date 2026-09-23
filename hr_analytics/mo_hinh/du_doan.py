"""Dự báo nguy cơ nghỉ việc: đơn lẻ, hàng loạt, what-if."""

from __future__ import annotations

from typing import Any

import pandas as pd

from xu_ly.tien_ich import LEAVE_LABEL, STAY_LABEL


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


def predict_attrition_batch(
    model,
    df: pd.DataFrame,
    feature_columns: list[str],
    *,
    id_col: str | None = None,
    keep_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Chấm điểm hàng loạt — trả DataFrame có xác suất + risk band."""
    missing = [c for c in feature_columns if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset batch thiếu feature: {missing[:8]}")

    X = df[feature_columns].copy()
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[:, 1]
    else:
        proba = model.predict(X).astype(float)

    out = pd.DataFrame(index=df.index)
    if id_col and id_col in df.columns:
        out[id_col] = df[id_col].values
    for col in keep_cols or []:
        if col in df.columns and col not in out.columns:
            out[col] = df[col].values

    out["probability"] = proba
    out["probability_pct"] = (proba * 100).round(1)
    out["risk_band"] = [risk_band(float(p)) for p in proba]
    out["prediction"] = [
        LEAVE_LABEL if float(p) >= 0.5 else STAY_LABEL for p in proba
    ]
    out = out.sort_values("probability", ascending=False).reset_index(drop=True)
    return out


def summarize_batch_risk(scored: pd.DataFrame) -> dict[str, Any]:
    """Tóm tắt phân bố rủi ro từ kết quả batch."""
    if scored is None or scored.empty:
        return {"total": 0, "High": 0, "Medium": 0, "Low": 0, "high_rate": 0.0}
    counts = scored["risk_band"].value_counts().to_dict()
    total = int(len(scored))
    high = int(counts.get("High", 0))
    return {
        "total": total,
        "High": high,
        "Medium": int(counts.get("Medium", 0)),
        "Low": int(counts.get("Low", 0)),
        "high_rate": round(100.0 * high / total, 1) if total else 0.0,
        "avg_probability_pct": round(float(scored["probability_pct"].mean()), 1),
    }


def simulate_what_if(
    model,
    base_features: dict[str, Any],
    changes: dict[str, Any],
    feature_columns: list[str],
) -> dict[str, Any]:
    """So sánh xác suất trước/sau khi thay đổi một số feature."""
    baseline = predict_attrition(model, base_features, feature_columns=feature_columns)
    scenario = dict(base_features)
    applied: dict[str, Any] = {}
    for key, value in changes.items():
        if key not in feature_columns:
            continue
        if value is None or value == "" or str(value).strip() == "":
            continue
        # Giữ kiểu số nếu gốc là số
        raw = base_features.get(key)
        try:
            if isinstance(raw, (int, float)) or (
                isinstance(raw, str) and str(raw).replace(".", "", 1).replace("-", "", 1).isdigit()
            ):
                scenario[key] = float(str(value).replace(",", ""))
            else:
                scenario[key] = value
        except (TypeError, ValueError):
            scenario[key] = value
        applied[key] = scenario[key]

    after = predict_attrition(model, scenario, feature_columns=feature_columns)
    delta_pp = round(after["probability_pct"] - baseline["probability_pct"], 1)
    if delta_pp < -0.5:
        effect = "Giảm rủi ro"
    elif delta_pp > 0.5:
        effect = "Tăng rủi ro"
    else:
        effect = "Hầu như không đổi"

    return {
        "baseline": baseline,
        "after": after,
        "delta_pp": delta_pp,
        "effect": effect,
        "applied_changes": applied,
        "scenario_features": scenario,
        "message": (
            f"Trước: {baseline['probability_pct']}% ({baseline['risk_band']}) → "
            f"Sau: {after['probability_pct']}% ({after['risk_band']}) · "
            f"Δ = {delta_pp:+.1f} điểm phần trăm ({effect})."
        ),
    }
