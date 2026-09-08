"""Sinh khuyến nghị HR từ insights (có bằng chứng)."""

from __future__ import annotations

from typing import Any

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
