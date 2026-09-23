"""Giải thích mô hình bằng SHAP (global + local)."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from xu_ly.chuan_bi_du_lieu import get_feature_names_from_preprocessor

# Giới hạn mẫu nền để SHAP chạy nhanh trên desktop
_DEFAULT_BG_SIZE = 120
_DEFAULT_GLOBAL_SIZE = 80


def _positive_class_shap(shap_values: Any) -> np.ndarray:
    """Lấy SHAP của lớp dương (nghỉ việc = 1) từ nhiều định dạng shap."""
    if isinstance(shap_values, list):
        # [class0, class1]
        arr = shap_values[1] if len(shap_values) > 1 else shap_values[0]
        return np.asarray(arr)
    arr = np.asarray(shap_values)
    if arr.ndim == 3:
        # (n_samples, n_features, n_classes)
        return arr[:, :, 1] if arr.shape[-1] > 1 else arr[:, :, 0]
    return arr


def _transform_matrix(
    pipeline,
    X: pd.DataFrame,
    numeric_features: list[str],
    categorical_features: list[str],
) -> tuple[np.ndarray, list[str]]:
    preprocessor = pipeline.named_steps["preprocess"]
    X_t = preprocessor.transform(X)
    if hasattr(X_t, "toarray"):
        X_t = X_t.toarray()
    X_t = np.asarray(X_t, dtype=float)
    names = get_feature_names_from_preprocessor(
        preprocessor, numeric_features, categorical_features
    )
    if len(names) != X_t.shape[1]:
        names = [f"feature_{i}" for i in range(X_t.shape[1])]
    return X_t, names


def _make_explainer(model, X_bg: np.ndarray):
    """TreeExplainer cho RF; fallback Explainer chung."""
    import shap

    model_name = type(model).__name__.lower()
    if "forest" in model_name or "tree" in model_name or "boost" in model_name:
        return shap.TreeExplainer(model)
    try:
        return shap.Explainer(model.predict_proba, X_bg)
    except Exception:  # noqa: BLE001
        return shap.Explainer(model.predict, X_bg)


def _sample_frame(X: pd.DataFrame, n: int, random_state: int = 42) -> pd.DataFrame:
    if len(X) <= n:
        return X.copy()
    return X.sample(n=n, random_state=random_state)


def explain_global_shap(
    pipeline,
    X: pd.DataFrame,
    numeric_features: list[str],
    categorical_features: list[str],
    *,
    top_n: int = 15,
    max_samples: int = _DEFAULT_GLOBAL_SIZE,
    background_size: int = _DEFAULT_BG_SIZE,
) -> dict[str, Any]:
    """SHAP toàn cục: mean(|SHAP|) theo feature đã encode."""
    X_use = _sample_frame(X, max_samples)
    X_bg = _sample_frame(X, min(background_size, len(X)))
    X_t, names = _transform_matrix(pipeline, X_use, numeric_features, categorical_features)
    X_bg_t, _ = _transform_matrix(pipeline, X_bg, numeric_features, categorical_features)

    model = pipeline.named_steps["model"]
    explainer = _make_explainer(model, X_bg_t)
    raw = explainer.shap_values(X_t)
    sv = _positive_class_shap(raw)
    if sv.ndim == 1:
        sv = sv.reshape(1, -1)

    mean_abs = np.mean(np.abs(sv), axis=0)
    fi = (
        pd.DataFrame({"Feature": names, "mean_|SHAP|": mean_abs})
        .sort_values("mean_|SHAP|", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )
    narrative = build_global_narrative(fi)
    return {
        "importance": fi,
        "feature_names": names,
        "n_samples": int(len(X_use)),
        "method": type(explainer).__name__,
        "narrative": narrative,
    }


def explain_local_shap(
    pipeline,
    employee_features: dict[str, Any] | pd.DataFrame,
    feature_columns: list[str],
    numeric_features: list[str],
    categorical_features: list[str],
    X_background: pd.DataFrame,
    *,
    top_n: int = 10,
    background_size: int = _DEFAULT_BG_SIZE,
) -> dict[str, Any]:
    """SHAP cục bộ cho một nhân viên — yếu tố đẩy tăng/giảm nguy cơ nghỉ việc."""
    if isinstance(employee_features, dict):
        row = pd.DataFrame([employee_features])[feature_columns]
    else:
        row = employee_features[feature_columns].copy()

    X_bg = _sample_frame(X_background[feature_columns], min(background_size, len(X_background)))
    X_row_t, names = _transform_matrix(pipeline, row, numeric_features, categorical_features)
    X_bg_t, _ = _transform_matrix(pipeline, X_bg, numeric_features, categorical_features)

    model = pipeline.named_steps["model"]
    explainer = _make_explainer(model, X_bg_t)
    raw = explainer.shap_values(X_row_t)
    sv = _positive_class_shap(raw)
    if sv.ndim == 2:
        sv = sv[0]

    base_value = None
    if hasattr(explainer, "expected_value"):
        ev = explainer.expected_value
        if isinstance(ev, (list, np.ndarray)):
            arr = np.asarray(ev).ravel()
            base_value = float(arr[1] if arr.size > 1 else arr[0])
        else:
            base_value = float(ev)

    contrib = (
        pd.DataFrame({
            "Feature": names,
            "SHAP": sv.astype(float),
            "AbsSHAP": np.abs(sv).astype(float),
            "Direction": np.where(sv >= 0, "Tăng rủi ro", "Giảm rủi ro"),
        })
        .sort_values("AbsSHAP", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )
    narrative = build_local_narrative(contrib, base_value=base_value)
    return {
        "contributions": contrib,
        "base_value": base_value,
        "feature_names": names,
        "shap_vector": sv,
        "method": type(explainer).__name__,
        "narrative": narrative,
        "top_increase": contrib[contrib["SHAP"] > 0].head(5).to_dict("records"),
        "top_decrease": contrib[contrib["SHAP"] < 0].head(5).to_dict("records"),
    }


def build_global_narrative(importance: pd.DataFrame, top_k: int = 5) -> str:
    if importance is None or importance.empty:
        return "Chưa đủ dữ liệu để giải thích SHAP toàn cục."
    tops = importance.head(top_k)
    parts = [
        f"{r['Feature']} (mean|SHAP|={r['mean_|SHAP|']:.4f})"
        for _, r in tops.iterrows()
    ]
    return (
        "Theo SHAP toàn cục, các yếu tố ảnh hưởng mạnh nhất tới dự báo nghỉ việc là: "
        + "; ".join(parts)
        + ". Giá trị mean(|SHAP|) càng cao thì feature càng quan trọng trên mẫu đã giải thích."
    )


def build_local_narrative(
    contributions: pd.DataFrame,
    *,
    base_value: float | None = None,
    top_k: int = 3,
) -> str:
    if contributions is None or contributions.empty:
        return "Không tính được đóng góp SHAP cho cá nhân này."
    up = contributions[contributions["SHAP"] > 0].head(top_k)
    down = contributions[contributions["SHAP"] < 0].head(top_k)
    lines: list[str] = []
    if base_value is not None:
        lines.append(f"Giá trị kỳ vọng nền của mô hình (base value) ≈ {base_value:.4f}.")
    if not up.empty:
        lines.append(
            "Yếu tố làm tăng nguy cơ nghỉ việc: "
            + "; ".join(f"{r['Feature']} (+{r['SHAP']:.4f})" for _, r in up.iterrows())
            + "."
        )
    if not down.empty:
        lines.append(
            "Yếu tố giúp giảm nguy cơ: "
            + "; ".join(f"{r['Feature']} ({r['SHAP']:.4f})" for _, r in down.iterrows())
            + "."
        )
    lines.append(
        "SHAP dương đẩy xác suất nghỉ việc lên; SHAP âm kéo xác suất xuống so với mức nền."
    )
    return " ".join(lines)


def plot_global_shap(importance: pd.DataFrame, title: str = "SHAP toàn cục (mean |SHAP|)"):
    fig, ax = plt.subplots(figsize=(9, 6))
    plot_df = importance.sort_values("mean_|SHAP|", ascending=True)
    ax.barh(plot_df["Feature"], plot_df["mean_|SHAP|"], color="#1F8A70")
    ax.set_title(title)
    ax.set_xlabel("mean(|SHAP|)")
    ax.set_ylabel("Feature")
    fig.tight_layout()
    return fig


def plot_local_shap(contributions: pd.DataFrame, title: str = "SHAP cục bộ — giải thích dự báo"):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    plot_df = contributions.sort_values("SHAP", ascending=True)
    colors = ["#C0392B" if v > 0 else "#1E8449" for v in plot_df["SHAP"]]
    ax.barh(plot_df["Feature"], plot_df["SHAP"], color=colors)
    ax.axvline(0, color="#5B6E78", linewidth=0.8)
    ax.set_title(title)
    ax.set_xlabel("SHAP value (lớp nghỉ việc)")
    ax.set_ylabel("Feature")
    fig.tight_layout()
    return fig


def shap_section_for_report(
    *,
    global_result: dict[str, Any] | None = None,
    local_result: dict[str, Any] | None = None,
    prediction: dict[str, Any] | None = None,
    model_name: str | None = None,
) -> dict[str, Any]:
    """Đóng gói nội dung SHAP để nhúng PDF/Word."""
    section: dict[str, Any] = {
        "title": "Giải thích mô hình (SHAP)",
        "model_name": model_name,
        "prediction": prediction,
        "global_narrative": None,
        "local_narrative": None,
        "global_top": [],
        "local_top": [],
    }
    if global_result:
        section["global_narrative"] = global_result.get("narrative")
        fi = global_result.get("importance")
        if fi is not None and not getattr(fi, "empty", True):
            section["global_top"] = [
                {"feature": r["Feature"], "value": float(r["mean_|SHAP|"])}
                for _, r in fi.head(10).iterrows()
            ]
    if local_result:
        section["local_narrative"] = local_result.get("narrative")
        contrib = local_result.get("contributions")
        if contrib is not None and not getattr(contrib, "empty", True):
            section["local_top"] = [
                {
                    "feature": r["Feature"],
                    "shap": float(r["SHAP"]),
                    "direction": r["Direction"],
                }
                for _, r in contrib.head(10).iterrows()
            ]
    return section
