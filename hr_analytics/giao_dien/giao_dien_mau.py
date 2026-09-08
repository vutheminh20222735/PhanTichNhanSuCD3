"""Design system — PeopleRisk AI Desktop (mockup-aligned)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Theme:
    brand: str = "#0B3A4A"
    brand_deep: str = "#072A36"
    brand_soft: str = "#123F4F"
    accent: str = "#1F8A70"
    accent_hover: str = "#18725C"
    accent_soft: str = "#E6F5F0"
    bg: str = "#F4F7F9"
    surface: str = "#FFFFFF"
    surface_alt: str = "#EEF2F5"
    border: str = "#D5DEE5"
    sidebar_text: str = "#D5E4EA"
    sidebar_muted: str = "#8FA9B4"
    text: str = "#1A2B34"
    text_secondary: str = "#5B6E78"
    text_muted: str = "#84949C"
    danger: str = "#C0392B"
    danger_soft: str = "#FDECEA"
    warning: str = "#B9770E"
    warning_soft: str = "#FEF5E7"
    success: str = "#1E8449"
    success_soft: str = "#E8F8F0"
    info: str = "#2471A3"
    info_soft: str = "#EAF4FB"
    font_family: str = "Noto Sans"
    font_fallback: str = "DejaVu Sans"


THEME = Theme()

NAV_SECTIONS = [
    (
        "DATA",
        [
            ("upload", "Nhập Dataset", "Upload CSV · chọn target"),
            ("dashboard", "Tổng quan", "KPI động theo dataset"),
            ("data", "Dữ liệu", "Dataset profile"),
            ("quality", "Chất lượng dữ liệu", "Quality score · làm sạch"),
        ],
    ),
    (
        "ANALYTICS",
        [
            ("eda", "Phân tích EDA", "Thống kê & quan hệ"),
            ("viz", "Trực quan hóa", "Biểu đồ theo cột thật"),
        ],
    ),
    (
        "AI / ML",
        [
            ("model", "Mô hình AI", "Classification · Regression"),
            ("predict", "Dự báo rủi ro", "Form động theo feature"),
        ],
    ),
    (
        "RESULTS",
        [
            ("insights", "Insights", "Từ dữ liệu hiện tại"),
            ("recs", "Khuyến nghị", "Dựa trên insight"),
        ],
    ),
]

FILTER_ROLE_LABELS = {
    "department": "Phòng ban",
    "location": "Địa điểm",
    "gender": "Giới tính",
    "overtime": "Làm thêm giờ",
    "contract": "Hợp đồng",
    "job_level": "Cấp bậc",
    "job_role": "Vị trí",
    "education": "Học vấn",
}
