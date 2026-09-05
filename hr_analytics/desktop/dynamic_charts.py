"""Matplotlib charts động theo schema roles — chỉ presentation."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.analysis.dynamic_eda import attrition_rate, attrition_rate_by_group, leave_mask
from src.analysis.correlation import compute_correlation_matrix
from src.utils import format_pct

_C_PRIMARY = "#0B3A4A"
_C_ACCENT = "#1F8A70"
_C_DANGER = "#C0392B"
_C_SUCCESS = "#1E8449"
_C_GOLD = "#CA8A04"

sns.set_theme(style="ticks", font="DejaVu Sans", font_scale=0.85)
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "figure.facecolor": "#FFFFFF",
    "figure.dpi": 72,
    "savefig.dpi": 72,
    "path.simplify": True,
    "path.simplify_threshold": 0.2,
})


def chart_attrition_donut(df: pd.DataFrame, target: str):
    leave = int(leave_mask(df[target]).sum())
    stay = len(df) - leave
    fig, ax = plt.subplots(figsize=(4.2, 3.2))
    ax.pie(
        [leave, stay],
        labels=["Nghỉ việc", "Ở lại"],
        colors=[_C_DANGER, _C_SUCCESS],
        autopct="%1.1f%%",
        startangle=90,
        pctdistance=0.75,
        wedgeprops=dict(width=0.42, edgecolor="white"),
    )
    rate = leave / len(df) * 100 if len(df) else 0
    ax.text(0, 0.06, f"{rate:.1f}%", ha="center", fontsize=13, fontweight="bold", color=_C_DANGER)
    ax.text(0, -0.12, "Attrition", ha="center", fontsize=8, color="#5B6E78")
    ax.set_title("Tình trạng nghỉ việc")
    fig.tight_layout()
    return fig, f"{leave} nghỉ / {stay} ở lại — {format_pct(rate)}."


def chart_rate_by_category(df: pd.DataFrame, col: str, target: str, title: str):
    table = attrition_rate_by_group(df, col, target)
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    sns.barplot(data=table, y=col, x="attrition_rate", ax=ax, color=_C_DANGER)
    ax.set_title(title)
    ax.set_xlabel("Tỷ lệ nghỉ việc (%)")
    ax.set_ylabel(col)
    fig.tight_layout()
    overall = attrition_rate(df, target)
    top = table.iloc[0]
    note = (
        f"Cao nhất: `{top[col]}` = {format_pct(top['attrition_rate'])}. "
        f"Trung bình dataset: {format_pct(overall)}."
    )
    return fig, note, table


def chart_count_by_category(df: pd.DataFrame, col: str, title: str):
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    order = df[col].value_counts().index
    # Top 12 categories nếu quá nhiều — nhanh hơn, vẫn đúng số liệu nhóm top
    if len(order) > 12:
        order = order[:12]
        plot_df = df[df[col].isin(order)]
    else:
        plot_df = df
    sns.countplot(data=plot_df, y=col, order=order, ax=ax, color=_C_PRIMARY)
    ax.set_title(title)
    ax.set_xlabel("Số nhân viên")
    ax.set_ylabel(col)
    fig.tight_layout()
    top = order[0]
    n = int((df[col] == top).sum())
    return fig, f"`{top}` đông nhất: {n} ({n/len(df)*100:.1f}%)."


def chart_hist(df: pd.DataFrame, col: str, title: str, xlabel: str, color: str = _C_PRIMARY):
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    s = pd.to_numeric(df[col], errors="coerce").dropna()
    sns.histplot(s, bins=18, kde=False, ax=ax, color=color)
    mean_v = float(s.mean())
    ax.axvline(mean_v, color=_C_DANGER, linestyle="--", label=f"Mean={mean_v:.1f}")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Số lượng")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig, f"Trung bình {mean_v:.2f}; min={s.min():.1f}, max={s.max():.1f}."


def chart_boxplot_by_target(df: pd.DataFrame, value_col: str, target: str, title: str, ylabel: str):
    fig, ax = plt.subplots(figsize=(4.8, 3.2))
    plot = df[[target, value_col]].copy()
    plot[value_col] = pd.to_numeric(plot[value_col], errors="coerce")
    sns.boxplot(
        data=plot, x=target, y=value_col, hue=target, ax=ax,
        palette=[_C_SUCCESS, _C_DANGER], legend=False,
    )
    ax.set_title(title)
    ax.set_xlabel(target)
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    med = plot.groupby(target)[value_col].median()
    note = "Median — " + "; ".join(f"{k}: {v:.2f}" for k, v in med.items())
    return fig, note


def chart_correlation(df: pd.DataFrame, target: str | None = None):
    from src.data.schema_detector import map_binary_target
    num = df.select_dtypes(include=[np.number]).copy()
    if target and target in df.columns and target not in num.columns:
        try:
            num = num.copy()
            num["Target_Encoded"] = map_binary_target(df[target])
        except Exception:  # noqa: BLE001
            pass
    # Giới hạn số cột heatmap để render nhanh
    if num.shape[1] > 14:
        # ưu tiên cột có variance cao
        variances = num.var(numeric_only=True).sort_values(ascending=False)
        keep = list(variances.head(13).index)
        if "Target_Encoded" in num.columns and "Target_Encoded" not in keep:
            keep.append("Target_Encoded")
        num = num[keep]
    if num.shape[1] < 2:
        raise ValueError("Không đủ biến số để vẽ correlation.")
    corr = num.corr()
    fig, ax = plt.subplots(figsize=(7.5, 6))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(corr, mask=mask, cmap="RdBu_r", center=0, ax=ax, square=True, cbar_kws={"shrink": 0.7})
    ax.set_title("Correlation heatmap")
    fig.tight_layout()
    return fig, "Tương quan quan sát được — không suy diễn nhân quả."


def gallery_specs(df: pd.DataFrame, roles: dict[str, Any], target: str) -> list[dict[str, Any]]:
    """Danh sách builder (lazy) — chưa vẽ figure, để UI render từng cái."""
    specs: list[dict[str, Any]] = []

    def add(title: str, builder) -> None:
        specs.append({"title": title, "builder": builder})

    dept = roles.get("department")
    role = roles.get("job_role")
    age = roles.get("age")
    income = roles.get("income")
    ot = roles.get("overtime")
    sat = roles.get("job_satisfaction")
    tenure = roles.get("tenure")
    dist = roles.get("distance")

    if dept:
        add("1. Phân bố theo phòng ban", lambda: chart_count_by_category(df, dept, "Employee by Department"))
    if role:
        add("2. Phân bố theo vị trí", lambda: chart_count_by_category(df, role, "Employee by Job Role"))
    if age:
        add("3. Phân bố tuổi", lambda: chart_hist(df, age, "Age Distribution", "Tuổi"))
    if income:
        def _income_hist():
            tmp = df.copy()
            tmp["_inc"] = pd.to_numeric(tmp[income], errors="coerce") / 1_000_000
            return chart_hist(tmp, "_inc", "Monthly Income", "Triệu VND", _C_ACCENT)

        add("4. Phân bố thu nhập", _income_hist)
    add("5. Phân bố nghỉ việc", lambda: chart_attrition_donut(df, target))
    if dept:
        add("6. Attrition theo phòng ban", lambda: chart_rate_by_category(df, dept, target, "Attrition by Department")[:2])
    if ot:
        add("7. Attrition theo Overtime", lambda: chart_rate_by_category(df, ot, target, "Attrition by Overtime")[:2])
    if sat:
        add("8. Attrition theo hài lòng", lambda: chart_rate_by_category(df, sat, target, "Attrition by Satisfaction")[:2])
    if income:
        add("9. Thu nhập theo nghỉ việc", lambda: chart_boxplot_by_target(df, income, target, "Income by Attrition", income))
    if tenure:
        add("10. Thâm niên theo nghỉ việc", lambda: chart_boxplot_by_target(df, tenure, target, "Tenure by Attrition", tenure))
    if dist:
        add("11. Khoảng cách theo nghỉ việc", lambda: chart_boxplot_by_target(df, dist, target, "Distance by Attrition", dist))
    add("12. Correlation heatmap", lambda: chart_correlation(df, target))
    return specs
