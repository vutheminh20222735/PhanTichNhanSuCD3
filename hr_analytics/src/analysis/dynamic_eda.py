"""EDA động theo schema / target."""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.data.schema_detector import map_binary_target
from src.utils import format_pct, format_vnd


def leave_mask(series: pd.Series) -> pd.Series:
    return map_binary_target(series) == 1


def attrition_rate(df: pd.DataFrame, target: str) -> float:
    if df.empty or target not in df.columns:
        return 0.0
    return float(leave_mask(df[target]).mean() * 100)


def attrition_rate_by_group(df: pd.DataFrame, group_col: str, target: str) -> pd.DataFrame:
    if group_col not in df.columns or target not in df.columns or df.empty:
        return pd.DataFrame(columns=[group_col, "total", "leaving", "attrition_rate"])
    leave = leave_mask(df[target])
    g = (
        df.assign(_leave=leave.astype(int))
        .groupby(group_col, dropna=False)
        .agg(total=("_leave", "size"), leaving=("_leave", "sum"))
        .reset_index()
    )
    g["attrition_rate"] = (g["leaving"] / g["total"] * 100).round(2)
    return g.sort_values("attrition_rate", ascending=False).reset_index(drop=True)


def compute_kpis_dynamic(df: pd.DataFrame, roles: dict[str, Any], target: str) -> dict[str, Any]:
    total = len(df)
    if total == 0 or target not in df.columns:
        return {
            "total_employees": 0,
            "employees_staying": 0,
            "employees_leaving": 0,
            "attrition_rate": 0.0,
            "avg_age": None,
            "avg_income": None,
            "avg_years_company": None,
            "avg_job_satisfaction": None,
        }
    leaving = int(leave_mask(df[target]).sum())
    result: dict[str, Any] = {
        "total_employees": total,
        "employees_staying": total - leaving,
        "employees_leaving": leaving,
        "attrition_rate": round(leaving / total * 100, 2),
        "avg_age": None,
        "avg_income": None,
        "avg_years_company": None,
        "avg_job_satisfaction": None,
    }
    for key, out in [
        ("age", "avg_age"),
        ("income", "avg_income"),
        ("tenure", "avg_years_company"),
        ("job_satisfaction", "avg_job_satisfaction"),
    ]:
        col = roles.get(key)
        if col and col in df.columns:
            result[out] = round(float(pd.to_numeric(df[col], errors="coerce").mean()), 2)
    return result


def run_research_questions(df: pd.DataFrame, roles: dict[str, Any], target: str) -> dict[str, Any]:
    """RQ1–RQ8 động — bỏ qua RQ nếu thiếu cột."""
    out: dict[str, Any] = {}

    # RQ1
    leave = leave_mask(df[target])
    vc = df[target].value_counts()
    pct = df[target].value_counts(normalize=True) * 100
    table = pd.DataFrame({
        "Value": vc.index.astype(str),
        "count": vc.values,
        "percentage": [round(float(x), 2) for x in pct.values],
    })
    out["rq1"] = {
        "question": "Phân bố nghỉ việc như thế nào?",
        "table": table,
        "summary": (
            f"Trong {len(df)} nhân viên, tỷ lệ nghỉ việc là "
            f"{format_pct(float(leave.mean()*100))} "
            f"({int(leave.sum())} người)."
        ),
        "available": True,
    }

    def _group_rq(key: str, role: str, question: str) -> dict[str, Any]:
        col = roles.get(role)
        if not col or col not in df.columns:
            return {"question": question, "available": False, "summary": f"Dataset không có biến `{role}`."}
        table = attrition_rate_by_group(df, col, target)
        top = table.iloc[0]
        overall = attrition_rate(df, target)
        return {
            "question": question,
            "available": True,
            "table": table,
            "column": col,
            "summary": (
                f"`{top[col]}` có tỷ lệ nghỉ việc {format_pct(top['attrition_rate'])} "
                f"({int(top['leaving'])}/{int(top['total'])}). "
                f"Mức trung bình dataset: {format_pct(overall)}."
            ),
        }

    out["rq2"] = _group_rq("rq2", "department", "Phòng ban nào có tỷ lệ nghỉ việc cao?")
    out["rq3"] = _group_rq("rq3", "overtime", "Làm thêm giờ có liên quan đến nghỉ việc không?")
    out["rq4"] = _group_rq("rq4", "job_satisfaction", "Mức độ hài lòng có liên quan đến nghỉ việc không?")

    # RQ5 income
    income = roles.get("income")
    if income and income in df.columns:
        tmp = df[[target, income]].copy()
        tmp[income] = pd.to_numeric(tmp[income], errors="coerce")
        leave_m = tmp.loc[leave_mask(tmp[target]), income].mean()
        stay_m = tmp.loc[~leave_mask(tmp[target]), income].mean()
        table = (
            tmp.groupby(target)[income]
            .agg(["count", "mean", "median", "std"])
            .reset_index()
            .round(2)
        )
        out["rq5"] = {
            "question": "Thu nhập có liên quan đến nghỉ việc không?",
            "available": True,
            "table": table,
            "column": income,
            "summary": (
                f"Thu nhập TB nhóm nghỉ: {format_vnd(leave_m)}; "
                f"nhóm ở lại: {format_vnd(stay_m)}."
            ),
        }
    else:
        out["rq5"] = {"question": "Thu nhập…", "available": False, "summary": "Không có biến thu nhập."}

    # RQ6 tenure
    tenure = roles.get("tenure")
    if tenure and tenure in df.columns:
        tmp = df[[target, tenure]].copy()
        tmp[tenure] = pd.to_numeric(tmp[tenure], errors="coerce")
        table = tmp.groupby(target)[tenure].agg(["count", "mean", "median"]).reset_index().round(2)
        out["rq6"] = {
            "question": "Thâm niên có liên quan đến nghỉ việc không?",
            "available": True,
            "table": table,
            "column": tenure,
            "summary": (
                f"Median thâm niên — nghỉ: {tmp.loc[leave_mask(tmp[target]), tenure].median():.2f}; "
                f"ở lại: {tmp.loc[~leave_mask(tmp[target]), tenure].median():.2f}."
            ),
        }
    else:
        out["rq6"] = {"question": "Thâm niên…", "available": False, "summary": "Không có biến thâm niên."}

    # RQ7 distance
    dist = roles.get("distance")
    if dist and dist in df.columns:
        tmp = df[[target, dist]].copy()
        tmp[dist] = pd.to_numeric(tmp[dist], errors="coerce")
        table = tmp.groupby(target)[dist].agg(["count", "mean", "median"]).reset_index().round(2)
        out["rq7"] = {
            "question": "Khoảng cách nhà–công ty có liên quan đến nghỉ việc không?",
            "available": True,
            "table": table,
            "column": dist,
            "summary": (
                f"Khoảng cách TB — nghỉ: {tmp.loc[leave_mask(tmp[target]), dist].mean():.2f}; "
                f"ở lại: {tmp.loc[~leave_mask(tmp[target]), dist].mean():.2f}."
            ),
        }
    else:
        out["rq7"] = {"question": "Khoảng cách…", "available": False, "summary": "Không có biến khoảng cách."}

    # RQ8 performance/training
    parts = []
    perf = roles.get("performance")
    train = roles.get("training")
    rq8: dict[str, Any] = {
        "question": "Hiệu suất và đào tạo có liên quan đến nghỉ việc không?",
        "available": bool(perf or train),
    }
    if perf and perf in df.columns:
        rq8["performance"] = attrition_rate_by_group(df, perf, target).sort_values(perf)
        parts.append(
            f"Hiệu suất TB — nghỉ: {df.loc[leave_mask(df[target]), perf].mean():.2f}; "
            f"ở lại: {df.loc[~leave_mask(df[target]), perf].mean():.2f}."
        )
    if train and train in df.columns:
        rq8["training"] = attrition_rate_by_group(df, train, target).sort_values(train)
        parts.append(
            f"Đào tạo/năm TB — nghỉ: {df.loc[leave_mask(df[target]), train].mean():.2f}; "
            f"ở lại: {df.loc[~leave_mask(df[target]), train].mean():.2f}."
        )
    rq8["summary"] = " ".join(parts) if parts else "Không đủ biến hiệu suất/đào tạo."
    out["rq8"] = rq8
    return out
