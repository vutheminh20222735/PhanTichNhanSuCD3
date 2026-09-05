"""Insight + Recommendation engines động theo schema."""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.analysis.dynamic_eda import attrition_rate, attrition_rate_by_group, leave_mask
from src.utils import format_pct, format_vnd


def _sev(diff_pp: float) -> str:
    ad = abs(diff_pp)
    if ad >= 15:
        return "HIGH"
    if ad >= 5:
        return "MEDIUM"
    return "LOW"


def generate_insights_dynamic(
    df: pd.DataFrame,
    roles: dict[str, Any],
    target: str,
) -> list[dict[str, Any]]:
    if df.empty or target not in df.columns:
        return []

    insights: list[dict[str, Any]] = []
    overall = attrition_rate(df, target)
    insights.append({
        "group": "Overview",
        "title": "Tỷ lệ nghỉ việc tổng thể",
        "insight": (
            f"Trên {len(df)} nhân viên, tỷ lệ nghỉ việc là {format_pct(overall)} "
            f"({int(leave_mask(df[target]).sum())} người)."
        ),
        "evidence": f"{format_pct(overall)}",
        "difference": f"n={len(df)}",
        "severity": "HIGH" if overall >= 25 else ("MEDIUM" if overall >= 15 else "LOW"),
        "metrics": {"attrition_rate": overall, "n": len(df)},
        "supported": True,
    })

    ot = roles.get("overtime")
    if ot and ot in df.columns:
        table = attrition_rate_by_group(df, ot, target)
        yes_rows = table[table[ot].astype(str).str.lower().isin(["có", "yes", "y", "true", "1"])]
        no_rows = table[table[ot].astype(str).str.lower().isin(["không", "no", "n", "false", "0"])]
        if not yes_rows.empty and not no_rows.empty:
            ry = float(yes_rows.iloc[0]["attrition_rate"])
            rn = float(no_rows.iloc[0]["attrition_rate"])
            diff = ry - rn
            insights.append({
                "group": "Overtime",
                "title": "Làm thêm giờ và nghỉ việc",
                "insight": (
                    f"Nhóm làm thêm giờ có tỷ lệ nghỉ việc {format_pct(ry)}, "
                    f"nhóm không làm thêm là {format_pct(rn)}. "
                    f"Chênh lệch {diff:+.2f} điểm phần trăm."
                ),
                "evidence": f"{ry:.2f}% vs {rn:.2f}%",
                "difference": f"{diff:+.2f} điểm %",
                "severity": _sev(diff),
                "metrics": {"rate_yes": ry, "rate_no": rn, "diff": diff},
                "supported": True,
            })

    sat = roles.get("job_satisfaction")
    if sat and sat in df.columns:
        table = attrition_rate_by_group(df, sat, target).sort_values(sat)
        if len(table) >= 2:
            low, high = table.iloc[0], table.iloc[-1]
            diff = float(low["attrition_rate"]) - float(high["attrition_rate"])
            insights.append({
                "group": "Job Satisfaction",
                "title": "Hài lòng công việc",
                "insight": (
                    f"Mức hài lòng {low[sat]}: {format_pct(low['attrition_rate'])}; "
                    f"mức {high[sat]}: {format_pct(high['attrition_rate'])}."
                ),
                "evidence": f"{low['attrition_rate']:.2f}% vs {high['attrition_rate']:.2f}%",
                "difference": f"{diff:+.2f} điểm %",
                "severity": _sev(diff),
                "metrics": {
                    "low_rate": float(low["attrition_rate"]),
                    "high_rate": float(high["attrition_rate"]),
                },
                "supported": True,
            })

    income = roles.get("income")
    if income and income in df.columns:
        leave_m = float(pd.to_numeric(df.loc[leave_mask(df[target]), income], errors="coerce").mean())
        stay_m = float(pd.to_numeric(df.loc[~leave_mask(df[target]), income], errors="coerce").mean())
        diff = stay_m - leave_m
        insights.append({
            "group": "Monthly Income",
            "title": "Thu nhập và nghỉ việc",
            "insight": (
                f"Thu nhập TB nhóm nghỉ: {format_vnd(leave_m)}; "
                f"nhóm ở lại: {format_vnd(stay_m)}."
            ),
            "evidence": f"{format_vnd(leave_m)} vs {format_vnd(stay_m)}",
            "difference": format_vnd(abs(diff)),
            "severity": _sev((diff / stay_m * 100) if stay_m else 0),
            "metrics": {"leave_mean": leave_m, "stay_mean": stay_m, "diff": diff},
            "supported": True,
        })

    tenure = roles.get("tenure")
    if tenure and tenure in df.columns:
        leave_t = pd.to_numeric(df.loc[leave_mask(df[target]), tenure], errors="coerce")
        stay_t = pd.to_numeric(df.loc[~leave_mask(df[target]), tenure], errors="coerce")
        insights.append({
            "group": "Years at Company",
            "title": "Thâm niên tại công ty",
            "insight": (
                f"Nhóm nghỉ: mean {leave_t.mean():.2f} (median {leave_t.median():.2f}); "
                f"nhóm ở lại: mean {stay_t.mean():.2f} (median {stay_t.median():.2f})."
            ),
            "evidence": f"{leave_t.mean():.2f}y vs {stay_t.mean():.2f}y",
            "difference": f"{leave_t.mean()-stay_t.mean():+.2f} năm",
            "severity": _sev((stay_t.mean() - leave_t.mean()) * 5),
            "metrics": {"leave_mean": float(leave_t.mean()), "stay_mean": float(stay_t.mean())},
            "supported": True,
        })

    dept = roles.get("department")
    if dept and dept in df.columns:
        table = attrition_rate_by_group(df, dept, target)
        if not table.empty:
            top = table.iloc[0]
            diff = float(top["attrition_rate"]) - overall
            insights.append({
                "group": "Department",
                "title": "Phòng ban nổi bật",
                "insight": (
                    f"`{top[dept]}` có tỷ lệ {format_pct(top['attrition_rate'])} "
                    f"({int(top['leaving'])}/{int(top['total'])}), "
                    f"so với overall {format_pct(overall)}."
                ),
                "evidence": f"{top[dept]}: {top['attrition_rate']:.2f}%",
                "difference": f"{diff:+.2f} điểm % vs overall",
                "severity": _sev(diff),
                "metrics": {
                    "department": top[dept],
                    "rate": float(top["attrition_rate"]),
                    "overall": overall,
                },
                "supported": True,
            })

    role = roles.get("job_role")
    if role and role in df.columns:
        table = attrition_rate_by_group(df, role, target)
        solid = table[table["total"] >= 20] if not table.empty else table
        if solid.empty:
            solid = table
        if not solid.empty:
            top = solid.iloc[0]
            insights.append({
                "group": "Job Role",
                "title": "Vị trí công việc rủi ro",
                "insight": (
                    f"`{top[role]}` có tỷ lệ nghỉ việc {format_pct(top['attrition_rate'])} "
                    f"trên {int(top['total'])} nhân viên."
                ),
                "evidence": f"{top['attrition_rate']:.2f}%",
                "difference": f"n={int(top['total'])}",
                "severity": _sev(float(top["attrition_rate"]) - overall),
                "metrics": {"role": top[role], "rate": float(top["attrition_rate"])},
                "supported": True,
            })

    dist = roles.get("distance")
    if dist and dist in df.columns:
        leave_d = float(pd.to_numeric(df.loc[leave_mask(df[target]), dist], errors="coerce").mean())
        stay_d = float(pd.to_numeric(df.loc[~leave_mask(df[target]), dist], errors="coerce").mean())
        insights.append({
            "group": "Distance",
            "title": "Khoảng cách nhà – công ty",
            "insight": f"Khoảng cách TB nhóm nghỉ {leave_d:.2f}; nhóm ở lại {stay_d:.2f}.",
            "evidence": f"{leave_d:.2f} vs {stay_d:.2f} km",
            "difference": f"{leave_d-stay_d:+.2f} km",
            "severity": _sev((leave_d - stay_d) * 3),
            "metrics": {"leave_mean": leave_d, "stay_mean": stay_d},
            "supported": True,
        })

    perf = roles.get("performance")
    if perf and perf in df.columns:
        leave_p = float(pd.to_numeric(df.loc[leave_mask(df[target]), perf], errors="coerce").mean())
        stay_p = float(pd.to_numeric(df.loc[~leave_mask(df[target]), perf], errors="coerce").mean())
        insights.append({
            "group": "Performance",
            "title": "Đánh giá hiệu suất",
            "insight": f"Hiệu suất TB — nghỉ: {leave_p:.2f}; ở lại: {stay_p:.2f}.",
            "evidence": f"{leave_p:.2f} vs {stay_p:.2f}",
            "difference": f"{leave_p-stay_p:+.2f}",
            "severity": _sev((stay_p - leave_p) * 10),
            "metrics": {"leave_mean": leave_p, "stay_mean": stay_p},
            "supported": True,
        })

    return insights


_RULES = {
    "Overtime": (
        "Khối lượng làm thêm giờ",
        "Rà soát OT policy, phân bổ workload và cân bằng thời gian làm việc.",
        "Giảm attrition ở nhóm OT cao",
    ),
    "Job Satisfaction": (
        "Hài lòng công việc thấp",
        "Khảo sát engagement và cải thiện trải nghiệm công việc cho nhóm điểm thấp.",
        "Tăng retention qua engagement",
    ),
    "Monthly Income": (
        "Chênh lệch thu nhập",
        "Rà soát khung lương/thưởng theo vị trí và benchmark thị trường.",
        "Cân bằng compensation",
    ),
    "Years at Company": (
        "Thâm niên ngắn",
        "Tăng cường onboarding và mentoring giai đoạn đầu (0–2 năm).",
        "Giảm early attrition",
    ),
    "Department": (
        "Phòng ban rủi ro cao",
        "Exit interview chuyên sâu, kiểm tra quản lý trực tiếp và workload phòng ban.",
        "Ổn định phòng ban nóng",
    ),
    "Job Role": (
        "Vị trí rủi ro cao",
        "Phân tích career path và hỗ trợ phát triển cho vị trí có attrition cao.",
        "Giữ chân theo role",
    ),
    "Distance": (
        "Khoảng cách lớn",
        "Xem xét hybrid/flexible work hoặc hỗ trợ đi lại.",
        "Giảm ma sát đi lại",
    ),
    "Performance": (
        "Hiệu suất và nghỉ việc",
        "Đối chiếu đánh giá hiệu suất với lộ trình phát triển nghề nghiệp.",
        "Giữ nhân sự theo performance band",
    ),
    "Overview": (
        "Attrition tổng thể",
        "Thiết lập dashboard theo dõi attrition định kỳ và cảnh báo nhóm rủi ro.",
        "Giám sát liên tục",
    ),
}


def generate_recommendations_dynamic(insights: list[dict[str, Any]]) -> list[dict[str, str]]:
    recs: list[dict[str, str]] = []
    for ins in insights:
        if not ins.get("supported"):
            continue
        group = ins.get("group", "")
        rule = _RULES.get(group)
        if not rule:
            continue
        sev = ins.get("severity", "MEDIUM")
        if group != "Overview" and sev == "LOW":
            continue
        problem, action, goal = rule
        recs.append({
            "problem": problem,
            "evidence": ins.get("evidence") or ins.get("insight", "")[:160],
            "recommended_action": action,
            "priority": sev,
            "expected_goal": goal,
            "based_on": ins.get("title", group),
            "linked_insight": ins.get("insight", ""),
        })
    return recs
