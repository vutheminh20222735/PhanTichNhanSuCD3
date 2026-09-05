"""Đánh giá mô hình Classification và Regression."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src.preprocessing.transformer import get_feature_names_from_preprocessor


def evaluate_classifier(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str,
) -> dict[str, Any]:
    """Tính Accuracy, Precision, Recall, F1, ROC-AUC + confusion matrix."""
    y_pred = model.predict(X_test)
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
    else:
        y_proba = y_pred.astype(float)

    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    metrics = {
        "Model": model_name,
        "Accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "Precision": round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
        "Recall": round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
        "F1": round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
        "ROC-AUC": round(float(roc_auc_score(y_test, y_proba)), 4),
    }

    return {
        "metrics": metrics,
        "confusion_matrix": cm,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "y_pred": y_pred,
        "y_proba": y_proba,
    }


def evaluate_all_classifiers(
    models: dict[str, Any],
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict[str, Any]:
    """Đánh giá nhiều classifier và chọn model dựa trên metric thực tế."""
    results = {}
    rows = []
    for name, model in models.items():
        ev = evaluate_classifier(model, X_test, y_test, name)
        results[name] = ev
        rows.append(ev["metrics"])

    metrics_df = pd.DataFrame(rows)

    # Ưu tiên phát hiện nghỉ việc: Recall, sau đó F1, ROC-AUC
    ranked = metrics_df.sort_values(
        by=["Recall", "F1", "ROC-AUC"], ascending=False
    ).reset_index(drop=True)
    best_name = ranked.iloc[0]["Model"]
    best_row = ranked.iloc[0]

    selection_reason = (
        f"Dựa trên mục tiêu phát hiện nhân viên có nguy cơ nghỉ việc, "
        f"model được chọn là **{best_name}** vì có Recall = {best_row['Recall']:.4f}, "
        f"F1 = {best_row['F1']:.4f}, ROC-AUC = {best_row['ROC-AUC']:.4f} "
        f"(so sánh trên tập test). Không mặc định Random Forest luôn tốt hơn."
    )

    return {
        "results": results,
        "metrics_table": metrics_df,
        "ranked": ranked,
        "best_model_name": best_name,
        "selection_reason": selection_reason,
    }


def plot_confusion_matrix(cm: np.ndarray, model_name: str):
    """Vẽ confusion matrix."""
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.figure.colorbar(im, ax=ax, fraction=0.046)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Dự đoán: Ở lại (0)", "Dự đoán: Nghỉ (1)"])
    ax.set_yticklabels(["Thực tế: Ở lại (0)", "Thực tế: Nghỉ (1)"])
    ax.set_title(f"Confusion Matrix — {model_name}")
    ax.set_xlabel("Dự đoán")
    ax.set_ylabel("Thực tế")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black", fontsize=14)
    fig.tight_layout()
    return fig


def plot_roc_curves(
    eval_results: dict[str, dict[str, Any]],
    y_test: pd.Series,
):
    """Vẽ ROC Curve nhiều model trên cùng biểu đồ."""
    fig, ax = plt.subplots(figsize=(7, 5))
    for name, ev in eval_results.items():
        fpr, tpr, _ = roc_curve(y_test, ev["y_proba"])
        auc = ev["metrics"]["ROC-AUC"]
        ax.plot(fpr, tpr, label=f"{name} (AUC = {auc:.4f})")
    ax.plot([0, 1], [0, 1], "k--", label="Random baseline")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve — so sánh mô hình")
    ax.legend(loc="lower right")
    fig.tight_layout()
    return fig


def get_rf_feature_importance(
    rf_pipeline,
    numeric_features: list[str],
    categorical_features: list[str],
    top_n: int = 15,
) -> pd.DataFrame:
    """Feature importance từ Random Forest với đúng tên sau OneHot."""
    preprocessor = rf_pipeline.named_steps["preprocess"]
    model = rf_pipeline.named_steps["model"]
    names = get_feature_names_from_preprocessor(
        preprocessor, numeric_features, categorical_features
    )
    importances = model.feature_importances_
    if len(names) != len(importances):
        names = [f"feature_{i}" for i in range(len(importances))]

    fi = (
        pd.DataFrame({"Feature": names, "Importance": importances})
        .sort_values("Importance", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )
    return fi


def plot_feature_importance(fi: pd.DataFrame, title: str = "Top Feature Importance"):
    """Vẽ horizontal bar feature importance."""
    fig, ax = plt.subplots(figsize=(9, 6))
    plot_df = fi.sort_values("Importance", ascending=True)
    ax.barh(plot_df["Feature"], plot_df["Importance"], color="#2E86AB")
    ax.set_title(title)
    ax.set_xlabel("Importance")
    ax.set_ylabel("Feature")
    fig.tight_layout()
    return fig


def evaluate_regression(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str = "Linear Regression",
) -> dict[str, Any]:
    """Đánh giá regression: MAE, MSE, RMSE, R²."""
    y_pred = model.predict(X_test)
    mse = float(mean_squared_error(y_test, y_pred))
    metrics = {
        "Model": model_name,
        "MAE": round(float(mean_absolute_error(y_test, y_pred)), 2),
        "MSE": round(mse, 2),
        "RMSE": round(float(np.sqrt(mse)), 2),
        "R2": round(float(r2_score(y_test, y_pred)), 4),
    }
    return {"metrics": metrics, "y_pred": y_pred}
