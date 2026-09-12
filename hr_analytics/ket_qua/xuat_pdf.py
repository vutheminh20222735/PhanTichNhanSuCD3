"""Xuất báo cáo PeopleRisk AI ra PDF (hỗ trợ tiếng Việt)."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from fpdf import FPDF

# Font hệ thống có dấu tiếng Việt
_FONT_CANDIDATES = [
    Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    Path("/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf"),
    Path("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"),
]
_FONT_BOLD_CANDIDATES = [
    Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
    Path("/usr/share/fonts/truetype/noto/NotoSans-Bold.ttf"),
    Path("/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"),
]


def _first_existing(paths: list[Path]) -> Path | None:
    for p in paths:
        if p.is_file():
            return p
    return None


def _safe(text: Any) -> str:
    if text is None:
        return ""
    s = str(text).replace("\r\n", "\n").replace("\r", "\n")
    # fpdf multi_cell không thích ký tự điều khiển lạ
    return "".join(ch if ord(ch) >= 32 or ch == "\n" else " " for ch in s)


class ReportPDF(FPDF):
    def __init__(self) -> None:
        super().__init__(orientation="P", unit="mm", format="A4")
        self.set_auto_page_break(auto=True, margin=18)
        regular = _first_existing(_FONT_CANDIDATES)
        bold = _first_existing(_FONT_BOLD_CANDIDATES)
        if regular is None:
            raise FileNotFoundError(
                "Không tìm thấy font TTF hỗ trợ tiếng Việt (DejaVu/Noto/Liberation)."
            )
        self.add_font("Report", "", str(regular))
        self.add_font("Report", "B", str(bold or regular))
        self.set_margins(16, 16, 16)

    def header(self) -> None:
        self.set_font("Report", "B", 11)
        self.set_text_color(11, 58, 74)
        self.cell(0, 8, "PeopleRisk AI — Báo cáo phân tích nhân sự", align="L")
        self.ln(4)
        self.set_draw_color(31, 138, 112)
        self.set_line_width(0.4)
        self.line(16, self.get_y(), 194, self.get_y())
        self.ln(6)

    def footer(self) -> None:
        self.set_y(-14)
        self.set_font("Report", "", 8)
        self.set_text_color(120, 120, 120)
        self.cell(0, 8, f"Trang {self.page_no()}/{{nb}}", align="C")

    def section_title(self, title: str) -> None:
        self.ln(2)
        self.set_font("Report", "B", 13)
        self.set_text_color(11, 58, 74)
        self.cell(0, 8, _safe(title), new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(213, 222, 229)
        self.line(16, self.get_y(), 194, self.get_y())
        self.ln(3)

    def body(self, text: str, size: int = 10) -> None:
        self.set_x(self.l_margin)
        self.set_font("Report", "", size)
        self.set_text_color(26, 43, 52)
        self.multi_cell(self.epw, 5.5, _safe(text), new_x="LMARGIN", new_y="NEXT")
        self.ln(1)

    def bullet(self, text: str) -> None:
        self.set_x(self.l_margin)
        self.set_font("Report", "", 10)
        self.set_text_color(26, 43, 52)
        self.multi_cell(self.epw, 5.5, f"• {_safe(text)}", new_x="LMARGIN", new_y="NEXT")
        self.ln(0.5)

    def kv_row(self, label: str, value: Any) -> None:
        label_w = 52
        self.set_x(self.l_margin)
        self.set_font("Report", "B", 10)
        self.set_text_color(91, 110, 120)
        y0 = self.get_y()
        x0 = self.l_margin
        self.set_xy(x0, y0)
        self.cell(label_w, 6, _safe(str(label)))
        self.set_font("Report", "", 10)
        self.set_text_color(26, 43, 52)
        self.set_xy(x0 + label_w, y0)
        self.multi_cell(self.epw - label_w, 6, _safe(value), new_x="LMARGIN", new_y="NEXT")



def export_report_pdf(
    path: str | Path,
    *,
    dataset_name: str | None,
    target: str | None,
    n_rows: int,
    n_cols: int,
    filter_note: str,
    kpis: dict[str, Any],
    insights: list[dict[str, Any]],
    recommendations: list[dict[str, Any]],
    model_name: str | None = None,
    model_metrics: dict[str, Any] | None = None,
    metrics_table: Any = None,
    generated_at: datetime | None = None,
) -> Path:
    """Tạo file PDF báo cáo tổng hợp. Trả về đường dẫn đã lưu."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    when = generated_at or datetime.now()

    pdf = ReportPDF()
    pdf.alias_nb_pages()
    pdf.add_page()

    # --- Meta ---
    pdf.section_title("1. Thông tin báo cáo")
    pdf.kv_row("Thời điểm xuất", when.strftime("%d/%m/%Y %H:%M:%S"))
    pdf.kv_row("Dataset", dataset_name or "—")
    pdf.kv_row("Kích thước", f"{n_rows:,} dòng × {n_cols} cột".replace(",", "."))
    pdf.kv_row("Target", target or "—")
    pdf.kv_row("Bộ lọc đang áp dụng", filter_note or "Không lọc")
    if model_name:
        pdf.kv_row("Mô hình đang dùng", model_name)

    # --- KPI ---
    pdf.section_title("2. Chỉ số tổng quan (KPI)")
    if not kpis:
        pdf.body("Không có KPI.")
    else:
        mapping = [
            ("Tổng nhân sự", kpis.get("total_employees")),
            ("Đang làm việc", kpis.get("employees_staying")),
            ("Đã nghỉ việc", kpis.get("employees_leaving")),
            ("Tỷ lệ nghỉ việc (%)", kpis.get("attrition_rate")),
            ("Tuổi trung bình", kpis.get("avg_age")),
            ("Thu nhập trung bình", kpis.get("avg_income")),
            ("Thâm niên TB (năm)", kpis.get("avg_years_company")),
            ("Hài lòng công việc TB", kpis.get("avg_job_satisfaction")),
        ]
        for label, val in mapping:
            if val is None:
                continue
            pdf.kv_row(label, val)

    # --- Model ---
    pdf.section_title("3. Kết quả mô hình")
    if model_metrics:
        pdf.body(f"Model được chọn: {model_name or '—'}")
        for key in ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]:
            if key in model_metrics:
                try:
                    pdf.kv_row(key, f"{float(model_metrics[key]):.4f}")
                except (TypeError, ValueError):
                    pdf.kv_row(key, model_metrics[key])
    elif metrics_table is not None and hasattr(metrics_table, "empty") and not metrics_table.empty:
        pdf.body("Bảng so sánh mô hình:")
        cols = [c for c in metrics_table.columns if c in (
            "Model", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC"
        )] or list(metrics_table.columns)[:6]
        for _, row in metrics_table.iterrows():
            parts = [f"{c}={row[c]}" for c in cols if c in row.index]
            pdf.bullet(" | ".join(parts))
    else:
        pdf.body("Chưa huấn luyện mô hình — phần này trống.")

    # --- Insights ---
    pdf.section_title(f"4. Insights ({len(insights)})")
    if not insights:
        pdf.body("Không có insight.")
    else:
        for i, ins in enumerate(insights, 1):
            title = ins.get("title") or ins.get("group") or f"Insight {i}"
            sev = ins.get("severity") or ""
            pdf.set_font("Report", "B", 10)
            pdf.set_text_color(11, 58, 74)
            pdf.set_x(pdf.l_margin)
            pdf.multi_cell(
                pdf.epw, 5.5,
                _safe(f"{i}. {title}" + (f"  [{sev}]" if sev else "")),
                new_x="LMARGIN", new_y="NEXT",
            )
            pdf.body(ins.get("insight") or "", size=10)
            ev = ins.get("evidence")
            if ev:
                pdf.body(f"Bằng chứng: {ev}", size=9)
            pdf.ln(1)

    # --- Recommendations ---
    pdf.section_title(f"5. Khuyến nghị ({len(recommendations)})")
    if not recommendations:
        pdf.body("Không có khuyến nghị.")
    else:
        for i, rec in enumerate(recommendations, 1):
            based = rec.get("based_on") or rec.get("problem") or f"Khuyến nghị {i}"
            pri = rec.get("priority") or ""
            pdf.set_font("Report", "B", 10)
            pdf.set_text_color(11, 58, 74)
            pdf.set_x(pdf.l_margin)
            pdf.multi_cell(
                pdf.epw, 5.5,
                _safe(f"{i}. {based}" + (f"  [{pri}]" if pri else "")),
                new_x="LMARGIN", new_y="NEXT",
            )
            if rec.get("problem"):
                pdf.body(f"Vấn đề: {rec['problem']}", size=10)
            if rec.get("evidence"):
                pdf.body(f"Bằng chứng: {rec['evidence']}", size=9)
            if rec.get("recommended_action"):
                pdf.body(f"Hành động: {rec['recommended_action']}", size=10)
            if rec.get("expected_goal"):
                pdf.body(f"Mục tiêu: {rec['expected_goal']}", size=9)
            pdf.ln(1)

    # --- Kết luận ngắn ---
    pdf.section_title("6. Kết luận tóm tắt")
    rate = kpis.get("attrition_rate")
    lines = [
        f"Dataset «{dataset_name or '—'}» với {n_rows:,} nhân viên đang phân tích "
        f"(target: {target or '—'}).".replace(",", "."),
    ]
    if rate is not None:
        lines.append(f"Tỷ lệ nghỉ việc hiện tại khoảng {rate}%.")
    if model_name:
        lines.append(f"Mô hình dự báo đang dùng: {model_name}.")
    if insights:
        lines.append(f"Đã rút {len(insights)} insight và {len(recommendations)} khuyến nghị hành động.")
    else:
        lines.append("Chưa có đủ insight — kiểm tra lại dữ liệu hoặc bộ lọc.")
    lines.append(
        "Báo cáo được sinh tự động từ dữ liệu và bộ lọc đang áp dụng trong PeopleRisk AI; "
        "không thay thế phán đoán chuyên môn HR."
    )
    for line in lines:
        pdf.bullet(line)

    pdf.output(str(out))
    return out
