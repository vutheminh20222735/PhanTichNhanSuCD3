"""PeopleRisk AI — Desktop HR Analytics."""

from __future__ import annotations

import os
import sys
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-hr-desktop")
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import customtkinter as ctk
import joblib
import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import pandas as pd

from desktop.dynamic_charts import (
    chart_attrition_donut,
    chart_boxplot_by_target,
    chart_rate_by_category,
    gallery_specs,
)
from desktop.perf import AnalysisCache, debounce, df_content_id, filter_signature, tune_matplotlib_fast
from desktop.theme import FILTER_ROLE_LABELS, NAV_SECTIONS, THEME
from desktop.widgets import (
    ScrollableFrame,
    body_text,
    clear_frame,
    configure_treeview_style,
    embed_figure,
    font,
    insight_card,
    make_kpi_card,
    panel,
    primary_button,
    quality_status_card,
    risk_result_card,
    rq_block,
    safe_chart,
    secondary_button,
    section_title,
    show_dataframe,
    status_pill,
)
from src.analysis.descriptive import descriptive_statistics
from src.analysis.dynamic_eda import compute_kpis_dynamic, run_research_questions
from src.data.loader import load_dataset, load_uploaded_dataset
from src.data.schema_detector import build_schema, detect_target_candidates
from src.data.validator import validate_dataset
from src.insights.dynamic_insights import (
    generate_insights_dynamic,
    generate_recommendations_dynamic,
)
from src.modeling.evaluate import (
    evaluate_all_classifiers,
    evaluate_regression,
    get_rf_feature_importance,
    plot_confusion_matrix,
    plot_feature_importance,
    plot_roc_curves,
)
from src.modeling.predict import predict_attrition
from src.modeling.train import train_classification_models, train_salary_regression
from src.preprocessing.cleaner import clean_dataset, detect_outliers_iqr
from src.preprocessing.transformer import get_feature_columns
from src.utils import DEFAULT_DATA_PATH, MODELS_DIR, apply_filters, format_vnd, format_vnd_compact

tune_matplotlib_fast()
ctk.set_appearance_mode("light")
ctk.set_default_color_theme("green")


class PeopleRiskApp(ctk.CTk):
    def __init__(self) -> None:
        super().__init__()
        self.title("PeopleRisk AI — HR Analytics Desktop")
        self.geometry("1540x940")
        self.minsize(1200, 720)
        self.configure(fg_color=THEME.bg)

        # Active dataset state
        self.df_raw = load_dataset()
        self.dataset_name = Path(DEFAULT_DATA_PATH).name
        self.schema = build_schema(self.df_raw)
        self.target = self.schema["target"] or "NghiViec"
        self.schema = build_schema(self.df_raw, target=self.target)
        self.loaded_at = datetime.now()
        self.is_default = True

        # Pending upload
        self.pending_df: pd.DataFrame | None = None
        self.pending_name: str | None = None
        self.pending_schema: dict | None = None
        self.pending_target_var: ctk.StringVar | None = None

        # Model state
        self.train_result = None
        self.eval_result = None
        self.salary_eval = None
        self.model_status = "Not trained"
        self.filter_vars: dict[str, Any] = {}
        self.filter_state: dict[str, str] = {}  # giữ lựa chọn khi rebuild trang
        self.predict_vars: dict[str, Any] = {}
        self.predict_step = 0
        self.current_page = "dashboard"
        self._nav_lookup = {k: (lb, h) for _, items in NAV_SECTIONS for k, lb, h in items}
        self.sidebar_dataset_lbl = None
        self.cache = AnalysisCache()
        self._dataset_id = df_content_id(self.df_raw)
        self._page_gen = 0

        configure_treeview_style(self)
        self._build_shell()
        self.show_page("dashboard")

    # ============================================================ SHELL
    def _build_shell(self) -> None:
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self.sidebar = ctk.CTkFrame(self, width=240, corner_radius=0, fg_color=THEME.brand)
        self.sidebar.grid(row=0, column=0, sticky="nsw")
        self.sidebar.grid_propagate(False)

        brand = ctk.CTkFrame(self.sidebar, fg_color=THEME.brand_deep, height=86, corner_radius=0)
        brand.pack(fill="x")
        brand.pack_propagate(False)
        ctk.CTkLabel(brand, text="PeopleRisk AI", font=font(18, "bold"), text_color="#FFF").pack(
            anchor="w", padx=16, pady=(16, 0)
        )
        ctk.CTkLabel(
            brand, text="HR Analytics & Prediction", font=font(10), text_color=THEME.sidebar_muted
        ).pack(anchor="w", padx=16)

        self.nav_buttons: dict[str, ctk.CTkButton] = {}
        for section, items in NAV_SECTIONS:
            ctk.CTkLabel(
                self.sidebar, text=section, font=font(9, "bold"), text_color=THEME.sidebar_muted
            ).pack(anchor="w", padx=16, pady=(12, 3))
            for key, label, _ in items:
                btn = ctk.CTkButton(
                    self.sidebar, text=f"  {label}", anchor="w", height=34, corner_radius=8,
                    fg_color="transparent", hover_color=THEME.brand_soft,
                    text_color=THEME.sidebar_text, font=font(12),
                    command=lambda k=key: self.show_page(k),
                )
                btn.pack(fill="x", padx=10, pady=1)
                self.nav_buttons[key] = btn

        foot = ctk.CTkFrame(self.sidebar, fg_color=THEME.brand_deep, corner_radius=0)
        foot.pack(side="bottom", fill="x")
        ctk.CTkLabel(foot, text="CURRENT DATASET", font=font(9, "bold"), text_color=THEME.sidebar_muted).pack(
            anchor="w", padx=14, pady=(10, 2)
        )
        self.sidebar_dataset_lbl = ctk.CTkLabel(
            foot, text="", font=font(11), text_color="#FFF", justify="left", wraplength=200
        )
        self.sidebar_dataset_lbl.pack(anchor="w", padx=14)
        primary_button(foot, "Đổi Dataset", lambda: self.show_page("upload"), height=32).pack(
            fill="x", padx=12, pady=(8, 12)
        )
        self._refresh_sidebar_dataset()

        self.main = ctk.CTkFrame(self, fg_color=THEME.bg, corner_radius=0)
        self.main.grid(row=0, column=1, sticky="nsew")
        self.main.grid_columnconfigure(0, weight=1)
        self.main.grid_rowconfigure(1, weight=1)

        self.topbar = ctk.CTkFrame(self.main, height=58, fg_color=THEME.surface, corner_radius=0)
        self.topbar.grid(row=0, column=0, sticky="ew")
        self.topbar.grid_propagate(False)
        left = ctk.CTkFrame(self.topbar, fg_color="transparent")
        left.pack(side="left", padx=16)
        self.header_label = ctk.CTkLabel(left, text="", font=font(16, "bold"), text_color=THEME.text)
        self.header_label.pack(anchor="w", pady=(8, 0))
        self.subheader_label = ctk.CTkLabel(left, text="", font=font(11), text_color=THEME.text_secondary)
        self.subheader_label.pack(anchor="w")
        right = ctk.CTkFrame(self.topbar, fg_color="transparent")
        right.pack(side="right", padx=12)
        self.model_pill_host = ctk.CTkFrame(right, fg_color="transparent")
        self.model_pill_host.pack(side="left", padx=4)
        secondary_button(right, "Xuất Excel", self._export, width=100, height=30).pack(side="left", padx=3)
        primary_button(right, "Làm mới", self._refresh, width=90, height=30).pack(side="left", padx=3)

        self.content = ScrollableFrame(self.main)
        self.content.grid(row=1, column=0, sticky="nsew")

        self.statusbar = ctk.CTkFrame(self.main, height=28, fg_color=THEME.surface_alt, corner_radius=0)
        self.statusbar.grid(row=2, column=0, sticky="ew")
        self.status_label = ctk.CTkLabel(self.statusbar, text="", font=font(10), text_color=THEME.text_muted)
        self.status_label.pack(side="left", padx=12)
        self.footer_right = ctk.CTkLabel(self.statusbar, text="", font=font(10), text_color=THEME.text_secondary)
        self.footer_right.pack(side="right", padx=12)

    def _refresh_sidebar_dataset(self) -> None:
        n, m = self.df_raw.shape
        txt = f"{self.dataset_name}\n{n:,} × {m}\nTarget: {self.target}"
        if self.sidebar_dataset_lbl:
            self.sidebar_dataset_lbl.configure(text=txt)

    def _set_header(self, key: str) -> None:
        label, hint = self._nav_lookup.get(key, ("", ""))
        self.header_label.configure(text=label)
        self.subheader_label.configure(text=hint)
        n, m = self.df_raw.shape
        model = self.eval_result["best_model_name"] if self.eval_result else self.model_status
        self.status_label.configure(
            text=f"PeopleRisk AI  ·  {self.loaded_at.strftime('%d/%m/%Y %H:%M')}"
        )
        self.footer_right.configure(
            text=f"{self.dataset_name}  ·  {n:,}×{m}  ·  Target:{self.target}  ·  Model:{model}"
        )
        clear_frame(self.model_pill_host)
        kind = "success" if self.eval_result else "warning"
        status_pill(self.model_pill_host, f"Model: {model}", kind).pack()

    def _highlight(self, key: str) -> None:
        for k, btn in self.nav_buttons.items():
            if k == key:
                btn.configure(fg_color=THEME.accent, text_color="#FFF", font=font(12, "bold"))
            else:
                btn.configure(fg_color="transparent", text_color=THEME.sidebar_text, font=font(12))

    def show_page(self, key: str) -> None:
        self.current_page = key
        self._page_gen += 1
        self._highlight(key)
        self._set_header(key)
        clear_frame(self.content)
        plt.close("all")
        {
            "dashboard": self._page_dashboard,
            "data": self._page_data,
            "upload": self._page_upload,
            "quality": self._page_quality,
            "eda": self._page_eda,
            "viz": self._page_viz,
            "model": self._page_model,
            "predict": self._page_predict,
            "insights": self._page_insights,
            "recs": self._page_recs,
        }[key]()

    def _defer(self, delay_ms: int, fn) -> None:
        """Chạy fn sau delay; bỏ qua nếu đã đổi trang."""
        gen = self._page_gen

        def _run() -> None:
            if gen != self._page_gen:
                return
            try:
                fn()
            except Exception as exc:  # noqa: BLE001
                body_text(self.content, f"Lỗi render: {exc}", muted=True)

        self.after(delay_ms, _run)

    def _filter_sig(self) -> str:
        return filter_signature(self.filter_state)

    def _cached_insights(self, df: pd.DataFrame):
        key = self.cache.key("ins", self._dataset_id, self.target, self._filter_sig(), len(df))
        hit = self.cache.get(key)
        if hit is not None:
            return hit
        roles = self.schema["roles"]
        return self.cache.set(key, generate_insights_dynamic(df, roles, self.target))

    def _cached_kpis(self, df: pd.DataFrame):
        key = self.cache.key("kpi", self._dataset_id, self.target, len(df), self._filter_sig())
        hit = self.cache.get(key)
        if hit is not None:
            return hit
        return self.cache.set(key, compute_kpis_dynamic(df, self.schema["roles"], self.target))

    def _cached_rq(self, df: pd.DataFrame):
        key = self.cache.key("rq", self._dataset_id, self.target, len(df), self._filter_sig())
        hit = self.cache.get(key)
        if hit is not None:
            return hit
        return self.cache.set(key, run_research_questions(df, self.schema["roles"], self.target))

    def _refresh(self) -> None:
        self.loaded_at = datetime.now()
        self.show_page(self.current_page)

    def _clear_model_state(self) -> None:
        self.train_result = self.eval_result = self.salary_eval = None
        self.model_status = "Not trained"
        self.cache.clear()

    def _activate_dataset(self, df: pd.DataFrame, name: str, target: str, is_default: bool = False) -> None:
        if target not in df.columns:
            messagebox.showerror("Target", f"Cột target `{target}` không tồn tại.")
            return
        self.df_raw = df
        self.dataset_name = name
        self.target = target
        self.schema = build_schema(df, target=target)
        self.is_default = is_default
        self.loaded_at = datetime.now()
        self._dataset_id = df_content_id(df)
        self._clear_model_state()
        self.filter_vars.clear()
        self.filter_state.clear()
        self.pending_df = None
        self._refresh_sidebar_dataset()
        messagebox.showinfo("Dataset", f"Đã kích hoạt: {name}\nTarget: {target}")
        self.show_page("dashboard")

    # ============================================================ FILTERS
    def _draw_filters(self, on_change) -> None:
        cols = [c for c in (self.schema.get("filter_columns") or []) if c in self.df_raw.columns]
        if not cols:
            return

        on_change_debounced = debounce(self, 250, on_change)
        box = ctk.CTkFrame(
            self.content, fg_color=THEME.surface, corner_radius=10,
            border_width=1, border_color=THEME.border, height=78,
        )
        box.pack(fill="x", padx=6, pady=4)
        box.pack_propagate(False)
        inner = ctk.CTkFrame(box, fg_color="transparent")
        inner.pack(fill="both", expand=True, padx=8, pady=6)
        ctk.CTkLabel(inner, text="BỘ LỌC", font=font(9, "bold"), text_color=THEME.text_muted).pack(
            side="left", padx=(2, 8)
        )

        saved = dict(self.filter_state)
        self.filter_vars.clear()
        roles = self.schema.get("roles", {})
        role_of = {v: k for k, v in roles.items() if v}
        self._filters_ready = False

        for col in cols:
            cell = ctk.CTkFrame(inner, fg_color="transparent")
            cell.pack(side="left", padx=4)
            label = FILTER_ROLE_LABELS.get(role_of.get(col, ""), col)
            ctk.CTkLabel(cell, text=label, font=font(9), text_color=THEME.text_muted).pack(anchor="w")
            opts = ["Tất cả"] + sorted(
                {str(x) for x in self.df_raw[col].dropna().tolist()}, key=str
            )
            current = str(saved.get(col, "Tất cả"))
            if current not in opts:
                current = "Tất cả"
            var = ctk.StringVar(value=current)
            self.filter_vars[col] = var
            self.filter_state[col] = current

            menu = ctk.CTkOptionMenu(
                cell, values=opts, variable=var, width=140, height=28, font=font(11),
                fg_color=THEME.surface_alt, button_color=THEME.brand, text_color=THEME.text,
            )
            menu.pack()

            def _on_pick(choice: str, c: str = col, m=menu) -> None:
                self.filter_state[c] = str(choice)
                if getattr(self, "_filters_ready", False):
                    on_change_debounced()

            menu.configure(command=_on_pick)

        secondary_button(inner, "Xóa lọc", self._clear_filters, width=88, height=28).pack(
            side="right", padx=4, pady=(12, 0)
        )
        # Bật sau 1 tick — tránh OptionMenu gọi command lúc khởi tạo
        self.after(50, lambda: setattr(self, "_filters_ready", True))

    def _clear_filters(self) -> None:
        self.filter_state.clear()
        self.filter_vars.clear()
        self.show_page(self.current_page)

    def _active_filters(self) -> dict[str, str]:
        """Các lọc đang có hiệu lực (bỏ 'Tất cả')."""
        return {
            c: str(v)
            for c, v in self.filter_state.items()
            if str(v).strip() and str(v) != "Tất cả" and c in self.df_raw.columns
        }

    def _filtered(self) -> pd.DataFrame:
        return apply_filters(self.df_raw, self._active_filters())

    def _filter_caption(self, df: pd.DataFrame) -> None:
        active = self._active_filters()
        if active:
            roles = self.schema.get("roles", {})
            role_of = {v: k for k, v in roles.items() if v}
            parts = [
                f"{FILTER_ROLE_LABELS.get(role_of.get(c, ''), c)}={v}"
                for c, v in active.items()
            ]
            detail = " · ".join(parts)
        else:
            detail = "không lọc"
        body_text(
            self.content,
            f"Đang phân tích {len(df):,} / {len(self.df_raw):,} nhân viên  ·  {detail}  ·  {self.dataset_name}",
            muted=True,
        )

    def _export(self) -> None:
        path = filedialog.asksaveasfilename(
            defaultextension=".xlsx", filetypes=[("Excel", "*.xlsx")],
            initialfile=f"PeopleRisk_{datetime.now():%Y%m%d_%H%M}.xlsx",
        )
        if not path:
            return
        try:
            df = self._filtered()
            roles = self.schema["roles"]
            kpis = compute_kpis_dynamic(df, roles, self.target)
            insights = generate_insights_dynamic(df, roles, self.target)
            recs = generate_recommendations_dynamic(insights)
            with pd.ExcelWriter(path, engine="openpyxl") as w:
                pd.DataFrame([kpis]).to_excel(w, sheet_name="KPI", index=False)
                pd.DataFrame(insights).to_excel(w, sheet_name="Insights", index=False)
                pd.DataFrame(recs).to_excel(w, sheet_name="Recommendations", index=False)
                if self.eval_result is not None:
                    self.eval_result["metrics_table"].to_excel(w, sheet_name="Model", index=False)
            messagebox.showinfo("Xuất báo cáo", f"Đã lưu:\n{path}")
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Lỗi", str(exc))

    # ============================================================ DASHBOARD
    def _page_dashboard(self) -> None:
        root = self.content
        self._draw_filters(lambda: self.show_page("dashboard"))
        df = self._filtered()
        self._filter_caption(df)
        if df.empty:
            body_text(root, "Không còn bản ghi sau khi lọc.")
            return
        roles = self.schema["roles"]
        kpis = self._cached_kpis(df)

        row1 = ctk.CTkFrame(root, fg_color="transparent")
        row1.pack(fill="x", padx=4, pady=2)
        for i in range(4):
            row1.grid_columnconfigure(i, weight=1)
        make_kpi_card(row1, "Tổng nhân sự", f"{kpis['total_employees']:,}", "headcount", "brand").grid(row=0, column=0, sticky="nsew", padx=3)
        make_kpi_card(row1, "Đang làm việc", f"{kpis['employees_staying']:,}", "active", "success").grid(row=0, column=1, sticky="nsew", padx=3)
        make_kpi_card(row1, "Đã nghỉ việc", f"{kpis['employees_leaving']:,}", "left", "danger").grid(row=0, column=2, sticky="nsew", padx=3)
        make_kpi_card(row1, "Tỷ lệ nghỉ việc", f"{kpis['attrition_rate']:.2f}%", "attrition rate", "warning").grid(row=0, column=3, sticky="nsew", padx=3)

        row2 = ctk.CTkFrame(root, fg_color="transparent")
        row2.pack(fill="x", padx=4, pady=2)
        extras = []
        if kpis.get("avg_age") is not None:
            extras.append(("Tuổi TB", f"{kpis['avg_age']:.1f}", "năm", "info"))
        if kpis.get("avg_income") is not None:
            extras.append(("Thu nhập TB", format_vnd_compact(kpis["avg_income"]), format_vnd(kpis["avg_income"])+"/tháng", "accent"))
        if kpis.get("avg_years_company") is not None:
            extras.append(("Thâm niên TB", f"{kpis['avg_years_company']:.2f} năm", "tại công ty", "brand"))
        if kpis.get("avg_job_satisfaction") is not None:
            extras.append(("Hài lòng CV", f"{kpis['avg_job_satisfaction']:.2f}", "thang 1–5", "success"))
        for i in range(max(len(extras), 1)):
            row2.grid_columnconfigure(i, weight=1)
        for i, (t, v, s, tone) in enumerate(extras):
            make_kpi_card(row2, t, v, s, tone).grid(row=0, column=i, sticky="nsew", padx=3)

        section_title(root, "Employee Attrition Overview")
        g = ctk.CTkFrame(root, fg_color="transparent")
        g.pack(fill="x", padx=2)
        g.grid_columnconfigure(0, weight=1)
        g.grid_columnconfigure(1, weight=1)
        L = ctk.CTkFrame(g, fg_color="transparent")
        R = ctk.CTkFrame(g, fg_color="transparent")
        L.grid(row=0, column=0, sticky="nsew", padx=2)
        R.grid(row=0, column=1, sticky="nsew", padx=2)
        loading_l = ctk.CTkLabel(L, text="Đang tải biểu đồ…", font=font(11), text_color=THEME.text_muted)
        loading_l.pack(pady=40)
        loading_r = ctk.CTkLabel(R, text="Đang tải biểu đồ…", font=font(11), text_color=THEME.text_muted)
        loading_r.pack(pady=40)

        dept = roles.get("department")

        def _draw_left() -> None:
            loading_l.destroy()
            fig, note, err = safe_chart(lambda: chart_attrition_donut(df, self.target))
            embed_figure(L, fig, 240, "Active vs Left", note, err)
            if fig:
                plt.close(fig)

        def _draw_right() -> None:
            loading_r.destroy()
            if dept:
                fig, note, err = safe_chart(
                    lambda: chart_rate_by_category(df, dept, self.target, "Attrition by Department")
                )
                embed_figure(R, fig, 240, "Theo phòng ban", note, err)
                if fig:
                    plt.close(fig)
            else:
                body_text(R, "Không có cột phòng ban trong dataset.", muted=True)

        self._defer(10, _draw_left)
        self._defer(60, _draw_right)

        # Insights + model: defer nhẹ để KPI hiện trước
        rest = ctk.CTkFrame(root, fg_color="transparent")
        rest.pack(fill="x", padx=2, pady=4)

        def _draw_rest() -> None:
            insights = self._cached_insights(df)
            section_title(rest, "Key Risk Signals")
            signals = [
                i for i in insights
                if i.get("group") != "Overview" and i.get("severity") in ("HIGH", "MEDIUM")
            ]
            if not signals:
                body_text(rest, "Không có tín hiệu rủi ro HIGH/MEDIUM.", muted=True)
            else:
                for idx, ins in enumerate(signals[:4], 1):
                    insight_card(
                        rest, idx, ins["group"], ins["title"], ins["insight"],
                        evidence=ins.get("evidence"), difference=ins.get("difference"),
                        severity=ins.get("severity"),
                    )

            section_title(rest, "Model Performance")
            if self.eval_result:
                best = self.eval_result["best_model_name"]
                m = self.eval_result["results"][best]["metrics"]
                mr = ctk.CTkFrame(rest, fg_color="transparent")
                mr.pack(fill="x", padx=4)
                for i, k in enumerate(["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]):
                    mr.grid_columnconfigure(i, weight=1)
                    make_kpi_card(mr, k, f"{m[k]:.4f}", best, "accent" if k in ("Recall", "F1") else "brand").grid(
                        row=0, column=i, sticky="nsew", padx=3
                    )
            else:
                body_text(rest, "Chưa huấn luyện mô hình.", muted=True)

            section_title(rest, "Recommended Actions")
            recs = generate_recommendations_dynamic(insights)
            for idx, r in enumerate(recs[:3], 1):
                insight_card(
                    rest, idx, r["priority"], r["based_on"],
                    r["evidence"], recommendation=r["recommended_action"], severity=r["priority"],
                )

        self._defer(100, _draw_rest)

    # ============================================================ DATA
    def _page_data(self) -> None:
        root = self.content
        sch = self.schema
        row = ctk.CTkFrame(root, fg_color="transparent")
        row.pack(fill="x", padx=4, pady=6)
        for i in range(4):
            row.grid_columnconfigure(i, weight=1)
        make_kpi_card(row, "Records", f"{sch['n_rows']:,}", tone="brand").grid(row=0, column=0, sticky="nsew", padx=3)
        make_kpi_card(row, "Variables", f"{sch['n_cols']}", tone="accent").grid(row=0, column=1, sticky="nsew", padx=3)
        make_kpi_card(row, "Numeric", f"{sch['n_numeric']}", tone="info").grid(row=0, column=2, sticky="nsew", padx=3)
        make_kpi_card(row, "Categorical", f"{sch['n_categorical']}", tone="warning").grid(row=0, column=3, sticky="nsew", padx=3)

        row2 = ctk.CTkFrame(root, fg_color="transparent")
        row2.pack(fill="x", padx=4, pady=2)
        for i in range(3):
            row2.grid_columnconfigure(i, weight=1)
        make_kpi_card(row2, "Missing", str(sch["missing_total"]), "Good" if sch["missing_total"] == 0 else "Check", "success" if sch["missing_total"] == 0 else "warning").grid(row=0, column=0, sticky="nsew", padx=3)
        make_kpi_card(row2, "Duplicate", str(sch["duplicate_total"]), "Good" if sch["duplicate_total"] == 0 else "Check", "success" if sch["duplicate_total"] == 0 else "warning").grid(row=0, column=1, sticky="nsew", padx=3)
        make_kpi_card(row2, "Target", str(self.target), "Confirmed", "accent").grid(row=0, column=2, sticky="nsew", padx=3)

        section_title(root, "Data Dictionary")
        rows = []
        for col in self.df_raw.columns:
            role = "Target" if col == self.target else ("ID" if col == sch.get("id_col") else "Feature")
            rows.append({
                "Column": col,
                "Data Type": str(self.df_raw[col].dtype),
                "Non-null": int(self.df_raw[col].notna().sum()),
                "Null": int(self.df_raw[col].isna().sum()),
                "Unique": int(self.df_raw[col].nunique()),
                "Role": role,
            })
        show_dataframe(root, pd.DataFrame(rows), height=240, page_size=12, enable_search=True)
        section_title(root, "Sample Data (20 dòng)")
        show_dataframe(root, self.df_raw.head(20), height=220, page_size=10, enable_search=True)

    # ============================================================ UPLOAD
    def _page_upload(self) -> None:
        root = self.content
        box = panel(root, "Nhập dataset", "Chọn file CSV hoặc dùng bộ dữ liệu mẫu")
        box.pack(fill="x", padx=8, pady=10)

        btns = ctk.CTkFrame(box, fg_color="transparent")
        btns.pack(padx=12, pady=12, anchor="w")
        primary_button(btns, "Chọn file CSV", self._pick_csv, width=160).pack(side="left", padx=4)
        secondary_button(btns, "Sử dụng dataset mẫu", self._use_default, width=180).pack(side="left", padx=4)

        if self.pending_df is None:
            body_text(
                root,
                f"Đang dùng: {self.dataset_name} ({len(self.df_raw):,}×{self.df_raw.shape[1]})",
                muted=True,
            )
            return

        sch = self.pending_schema or build_schema(self.pending_df)
        section_title(root, "Phân tích Dataset đã tải")
        info = ctk.CTkFrame(root, fg_color="transparent")
        info.pack(fill="x", padx=4)
        for i in range(4):
            info.grid_columnconfigure(i, weight=1)
        make_kpi_card(info, "File", self.pending_name or "", tone="brand").grid(row=0, column=0, sticky="nsew", padx=3)
        make_kpi_card(info, "Records", f"{sch['n_rows']:,}", tone="accent").grid(row=0, column=1, sticky="nsew", padx=3)
        make_kpi_card(info, "Variables", f"{sch['n_cols']}", tone="info").grid(row=0, column=2, sticky="nsew", padx=3)
        make_kpi_card(info, "Missing / Dup", f"{sch['missing_total']} / {sch['duplicate_total']}", tone="success").grid(row=0, column=3, sticky="nsew", padx=3)

        make_kpi_card(info, "Numeric", f"{sch['n_numeric']}", tone="brand").grid(row=1, column=0, sticky="nsew", padx=3, pady=4)
        make_kpi_card(info, "Categorical", f"{sch['n_categorical']}", tone="accent").grid(row=1, column=1, sticky="nsew", padx=3, pady=4)

        candidates = sch.get("target_candidates") or detect_target_candidates(self.pending_df)
        section_title(root, "Chọn biến mục tiêu (Target)")
        if not candidates:
            body_text(root, "Không tìm thấy cột mục tiêu phù hợp — chọn thủ công.")
            candidates = list(self.pending_df.columns)
        self.pending_target_var = ctk.StringVar(value=candidates[0])
        ctk.CTkOptionMenu(
            root, values=candidates, variable=self.pending_target_var, width=280, height=34
        ).pack(anchor="w", padx=12, pady=6)

        section_title(root, "Preview")
        show_dataframe(root, self.pending_df.head(15), height=200, page_size=10)

        primary_button(root, "Xác nhận Dataset", self._confirm_pending, width=200).pack(
            anchor="w", padx=12, pady=14
        )

    def _pick_csv(self) -> None:
        path = filedialog.askopenfilename(filetypes=[("CSV", "*.csv"), ("All", "*.*")])
        if not path:
            return
        try:
            df = load_uploaded_dataset(path)
            if df.shape[1] < 2:
                messagebox.showerror("Dataset", "Dataset cần ít nhất 2 cột.")
                return
            self.pending_df = df
            self.pending_name = Path(path).name
            self.pending_schema = build_schema(df)
            self.show_page("upload")
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Lỗi CSV", str(exc))

    def _use_default(self) -> None:
        df = load_dataset()
        sch = build_schema(df)
        self._activate_dataset(df, Path(DEFAULT_DATA_PATH).name, sch["target"] or list(df.columns)[-1], True)

    def _confirm_pending(self) -> None:
        if self.pending_df is None or self.pending_target_var is None:
            return
        target = self.pending_target_var.get()
        self._activate_dataset(self.pending_df, self.pending_name or "upload.csv", target, False)

    # ============================================================ QUALITY
    def _page_quality(self) -> None:
        root = self.content
        validation = validate_dataset(self.df_raw)
        missing = validation["total_missing"]
        dup = validation["duplicates"]["duplicate_count"]
        inv = int(validation["invalid_numeric"]["Invalid Count"].sum())
        outliers = detect_outliers_iqr(self.df_raw)
        out_n = int(outliers["Outlier Count"].sum()) if not outliers.empty else 0
        # Quality score simple
        score = 100.0
        if len(self.df_raw):
            score -= min(40, missing / len(self.df_raw) * 100)
            score -= min(30, dup / len(self.df_raw) * 100)
            score -= min(20, inv / max(len(self.df_raw), 1) * 5)
        score = max(0, round(score, 1))
        label = "Excellent" if score >= 95 else ("Good" if score >= 80 else "Needs attention")

        row = ctk.CTkFrame(root, fg_color="transparent")
        row.pack(fill="x", padx=6, pady=10)
        quality_status_card(row, "Missing", missing, "0 Missing").pack(side="left", padx=6)
        quality_status_card(row, "Duplicate", dup, "0 Duplicate").pack(side="left", padx=6)
        quality_status_card(row, "Invalid", inv, "0 Invalid").pack(side="left", padx=6)
        qcard = ctk.CTkFrame(row, fg_color=THEME.success_soft if score >= 95 else THEME.warning_soft, corner_radius=10, width=180, height=88)
        qcard.pack(side="left", padx=6)
        qcard.pack_propagate(False)
        ctk.CTkLabel(qcard, text=f"Quality\n{score}%\n{label}", font=font(13, "bold"),
                     text_color=THEME.success if score >= 95 else THEME.warning).pack(expand=True)

        body_text(root, validation["missing_message"], muted=True)
        body_text(root, validation["duplicates"]["message"], muted=True)
        section_title(root, "Outlier IQR (không tự xóa)")
        show_dataframe(root, outliers, height=160, page_size=8)
        section_title(root, "Invalid numeric")
        show_dataframe(root, validation["invalid_numeric"], height=160, page_size=10)
        primary_button(root, "Chạy làm sạch & lưu processed", self._run_clean).pack(anchor="w", padx=10, pady=10)

    def _run_clean(self) -> None:
        _, report = clean_dataset(self.df_raw, drop_duplicates=True, save=True)
        messagebox.showinfo("Clean", f"Rows {report['rows_before']}→{report['rows_after']}\n{report.get('saved_path','')}")

    # ============================================================ EDA
    def _page_eda(self) -> None:
        root = self.content
        self._draw_filters(lambda: self.show_page("eda"))
        df = self._filtered()
        self._filter_caption(df)
        if df.empty:
            body_text(root, "Không còn bản ghi.")
            return
        roles = self.schema["roles"]
        rq = self._cached_rq(df)

        # Descriptive for known numeric roles
        num_cols = [c for c in self.schema["numeric"] if c in df.columns][:16]
        if num_cols:
            section_title(root, "Thống kê mô tả")
            show_dataframe(root, descriptive_statistics(df, num_cols), height=200, page_size=10)

        chart_map = {
            "rq1": lambda: chart_attrition_donut(df, self.target),
            "rq2": (lambda: chart_rate_by_category(df, roles["department"], self.target, "Attrition by Department")) if roles.get("department") else None,
            "rq3": (lambda: chart_rate_by_category(df, roles["overtime"], self.target, "Attrition by Overtime")) if roles.get("overtime") else None,
            "rq4": (lambda: chart_rate_by_category(df, roles["job_satisfaction"], self.target, "By Satisfaction")) if roles.get("job_satisfaction") else None,
            "rq5": (lambda: chart_boxplot_by_target(df, roles["income"], self.target, "Income", roles["income"])) if roles.get("income") else None,
            "rq6": (lambda: chart_boxplot_by_target(df, roles["tenure"], self.target, "Tenure", roles["tenure"])) if roles.get("tenure") else None,
            "rq7": (lambda: chart_boxplot_by_target(df, roles["distance"], self.target, "Distance", roles["distance"])) if roles.get("distance") else None,
        }

        keys = ["rq1", "rq2", "rq3", "rq4", "rq5", "rq6", "rq7", "rq8"]

        def _render_rq(i: int) -> None:
            if i >= len(keys):
                return
            key = keys[i]
            data = rq.get(key, {})
            box = rq_block(root, f"{key.upper()} — {data.get('question', '')}")
            if not data.get("available", True) and key != "rq1":
                body_text(box, data.get("summary", "Không đủ dữ liệu."), muted=True)
                self._defer(20, lambda j=i + 1: _render_rq(j))
                return
            builder = chart_map.get(key)
            if builder:
                fig, note, err = safe_chart(builder)
                embed_figure(box, fig, 230, "", note, err)
                if fig:
                    plt.close(fig)
            if isinstance(data.get("table"), pd.DataFrame) and not data["table"].empty:
                show_dataframe(box, data["table"], height=110, page_size=6)
            if "performance" in data and isinstance(data["performance"], pd.DataFrame):
                show_dataframe(box, data["performance"], height=110, page_size=6)
            if "training" in data and isinstance(data["training"], pd.DataFrame):
                show_dataframe(box, data["training"], height=110, page_size=6)
            body_text(box, data.get("summary", ""))
            self._defer(25, lambda j=i + 1: _render_rq(j))

        self._defer(15, lambda: _render_rq(0))

    # ============================================================ VIZ
    def _page_viz(self) -> None:
        root = self.content
        self._draw_filters(lambda: self.show_page("viz"))
        df = self._filtered()
        self._filter_caption(df)
        if df.empty:
            body_text(root, "Không còn bản ghi.")
            return

        status = ctk.CTkLabel(
            root, text="Đang tải gallery biểu đồ…", font=font(12), text_color=THEME.text_muted
        )
        status.pack(anchor="w", padx=12, pady=8)
        host = ctk.CTkFrame(root, fg_color="transparent")
        host.pack(fill="x", padx=2)

        specs = gallery_specs(df, self.schema["roles"], self.target)
        if not specs:
            status.configure(text="")
            body_text(host, "Không tạo được biểu đồ.")
            return
        status.configure(text=f"Gallery {len(specs)} biểu đồ — đang tải…")

        def _render_pair(start: int) -> None:
            if start >= len(specs):
                status.configure(text=f"Đã tải {len(specs)} biểu đồ")
                return
            row = ctk.CTkFrame(host, fg_color="transparent")
            row.pack(fill="x", padx=2, pady=2)
            row.grid_columnconfigure(0, weight=1)
            row.grid_columnconfigure(1, weight=1)
            for j in range(2):
                idx = start + j
                if idx >= len(specs):
                    break
                spec = specs[idx]
                cell = ctk.CTkFrame(row, fg_color="transparent")
                cell.grid(row=0, column=j, sticky="nsew", padx=2)
                fig, note, err = safe_chart(spec["builder"])
                embed_figure(cell, fig, 250, spec["title"], note, err)
                if fig:
                    plt.close(fig)
            status.configure(text=f"Đã tải {min(start + 2, len(specs))}/{len(specs)} biểu đồ…")
            self._defer(35, lambda s=start + 2: _render_pair(s))

        self._defer(20, lambda: _render_pair(0))

    # ============================================================ MODEL
    def _page_model(self) -> None:
        root = self.content
        intro = panel(root, "Phân loại nghỉ việc", f"Target: {self.target}  ·  Train/Test 80/20 (stratify)")
        intro.pack(fill="x", padx=6, pady=6)
        status = ctk.CTkLabel(intro, text="", font=font(11), text_color=THEME.text_muted)
        status.pack(anchor="w", padx=12)

        def train_cls() -> None:
            status.configure(text="Đang train Logistic Regression & Random Forest...")
            self.update_idletasks()
            try:
                tr = train_classification_models(self.df_raw, save=True, target=self.target)
                ev = evaluate_all_classifiers(tr["models"], tr["X_test"], tr["y_test"])
                self.train_result, self.eval_result = tr, ev
                self.model_status = ev["best_model_name"]
                meta_path = MODELS_DIR / "feature_meta.joblib"
                meta = joblib.load(meta_path) if meta_path.exists() else {}
                meta.update({
                    "best_model_name": ev["best_model_name"],
                    "target": self.target,
                    "feature_columns": tr["feature_columns"],
                })
                joblib.dump(meta, meta_path)
                self.show_page("model")
            except Exception as exc:  # noqa: BLE001
                messagebox.showerror("Train error", str(exc))

        def train_reg() -> None:
            status.configure(text="Đang train Salary Regression...")
            self.update_idletasks()
            try:
                sal = train_salary_regression(self.df_raw, save=True)
                self.salary_eval = evaluate_regression(sal["pipeline"], sal["X_test"], sal["y_test"])
                self.show_page("model")
            except Exception as exc:  # noqa: BLE001
                messagebox.showwarning("Regression", str(exc))

        btns = ctk.CTkFrame(intro, fg_color="transparent")
        btns.pack(anchor="w", padx=10, pady=8)
        primary_button(btns, "Huấn luyện phân loại", train_cls, width=180).pack(side="left", padx=3)
        secondary_button(btns, "Huấn luyện hồi quy thu nhập", train_reg, width=210).pack(side="left", padx=3)

        if self.eval_result is None:
            body_text(root, "Chưa có mô hình phân loại.", muted=True)
        else:
            best = self.eval_result["best_model_name"]
            m = self.eval_result["results"][best]["metrics"]
            section_title(root, f"Hiệu năng — {best}")
            mr = ctk.CTkFrame(root, fg_color="transparent")
            mr.pack(fill="x", padx=4)
            for i, k in enumerate(["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]):
                mr.grid_columnconfigure(i, weight=1)
                make_kpi_card(mr, k, f"{m[k]:.4f}", tone="accent" if k in ("Recall", "F1") else "brand").grid(
                    row=0, column=i, sticky="nsew", padx=3
                )
            section_title(root, "So sánh mô hình")
            show_dataframe(root, self.eval_result["metrics_table"], height=90, page_size=5)
            conclusion = panel(root, "Model được chọn")
            conclusion.pack(fill="x", padx=6, pady=6)
            body_text(conclusion, self.eval_result["selection_reason"].replace("**", ""))

            grid = ctk.CTkFrame(root, fg_color="transparent")
            grid.pack(fill="x")
            grid.grid_columnconfigure(0, weight=1)
            grid.grid_columnconfigure(1, weight=1)
            for idx, (name, ev) in enumerate(self.eval_result["results"].items()):
                cell = ctk.CTkFrame(grid, fg_color="transparent")
                cell.grid(row=0, column=idx, sticky="nsew", padx=2)
                fig = plot_confusion_matrix(ev["confusion_matrix"], name)
                embed_figure(cell, fig, 250, f"Confusion — {name}")
                plt.close(fig)

            fig = plot_roc_curves(self.eval_result["results"], self.train_result["y_test"])
            embed_figure(root, fig, 300, "ROC Curve")
            plt.close(fig)
            fi = get_rf_feature_importance(
                self.train_result["models"]["Random Forest"],
                self.train_result["numeric_features"],
                self.train_result["categorical_features"], 15,
            )
            show_dataframe(root, fi, height=220, page_size=15)
            fig = plot_feature_importance(fi)
            embed_figure(root, fig, 300, "Feature Importance")
            plt.close(fig)

        section_title(root, "Hồi quy thu nhập")
        if self.salary_eval is None:
            if not self.schema["roles"].get("income"):
                body_text(root, "Không có cột thu nhập.", muted=True)
        else:
            m = self.salary_eval["metrics"]
            show_dataframe(root, pd.DataFrame([m]), height=70, page_size=3)
            body_text(root, f"MAE ≈ {format_vnd(m['MAE'])} · RMSE ≈ {format_vnd(m['RMSE'])} · R²={m['R2']:.4f}")

    # ============================================================ PREDICT
    def _page_predict(self) -> None:
        root = self.content
        roles = self.schema["roles"]

        groups = [
            ("PERSONAL", ["age", "gender", "marital", "location"]),
            ("JOB", ["department", "job_role", "job_level", "contract", "travel", "education", "field"]),
            ("COMPENSATION", ["income", "salary_hike"]),
            ("WORK", ["overtime", "distance", "tenure", "total_experience", "num_companies",
                      "years_role", "years_promo", "years_manager"]),
            ("SATISFACTION", ["job_satisfaction", "env_satisfaction", "worklife", "involvement", "relationship"]),
            ("PERFORMANCE", ["performance", "training"]),
        ]

        form = ctk.CTkFrame(root, fg_color="transparent")
        form.pack(fill="x", padx=4, pady=4)
        # 3 columns layout
        cols_ui = [ctk.CTkFrame(form, fg_color="transparent") for _ in range(3)]
        for i, c in enumerate(cols_ui):
            form.grid_columnconfigure(i, weight=1)
            c.grid(row=0, column=i, sticky="nsew", padx=3)

        numeric, categorical = get_feature_columns(self.df_raw, target=self.target)
        self.predict_vars = {}
        # distribute groups across 3 columns
        for gi, (title, role_keys) in enumerate(groups):
            host = cols_ui[gi % 3]
            card = panel(host, title)
            card.pack(fill="x", pady=4)
            for rk in role_keys:
                col = roles.get(rk)
                if not col or col not in self.df_raw.columns:
                    continue
                if col in self.predict_vars:
                    continue
                cell = ctk.CTkFrame(card, fg_color="transparent")
                cell.pack(fill="x", padx=10, pady=3)
                ctk.CTkLabel(cell, text=col, font=font(10, "bold"), text_color=THEME.text_muted).pack(anchor="w")
                if col in numeric:
                    series = pd.to_numeric(self.df_raw[col], errors="coerce")
                    var = ctk.StringVar(value=str(int(round(float(series.median())))))
                    ctk.CTkEntry(cell, textvariable=var, height=28, fg_color=THEME.surface_alt).pack(fill="x")
                    self.predict_vars[col] = ("num", var)
                else:
                    opts = sorted(self.df_raw[col].dropna().astype(str).unique().tolist())
                    var = ctk.StringVar(value=opts[0] if opts else "")
                    ctk.CTkOptionMenu(cell, values=opts or [""], variable=var, height=28,
                                     fg_color=THEME.surface_alt, button_color=THEME.brand).pack(fill="x")
                    self.predict_vars[col] = ("cat", var)

        # fill remaining features with hidden defaults
        for col in numeric + categorical:
            if col in self.predict_vars:
                continue
            if col in numeric:
                series = pd.to_numeric(self.df_raw[col], errors="coerce")
                self.predict_vars[col] = ("num", ctk.StringVar(value=str(float(series.median()))))
            else:
                mode = self.df_raw[col].mode()
                self.predict_vars[col] = ("cat", ctk.StringVar(value=str(mode.iloc[0]) if len(mode) else ""))

        result_host = ctk.CTkFrame(root, fg_color="transparent")
        result_host.pack(fill="x", padx=4, pady=8)

        def run() -> None:
            try:
                model, feats, name = self._load_model()
                inputs = {}
                for col, (kind, var) in self.predict_vars.items():
                    raw = var.get()
                    inputs[col] = float(str(raw).replace(",", "")) if kind == "num" else raw
                # cast job level int-like
                jl = roles.get("job_level")
                if jl and jl in inputs:
                    try:
                        inputs[jl] = int(float(inputs[jl]))
                    except Exception:  # noqa: BLE001
                        pass
                result = predict_attrition(model, inputs, feature_columns=feats)
                clear_frame(result_host)
                risk_result_card(
                    result_host, result["probability_pct"], result["risk_band"],
                    result["prediction"], name,
                )
                # contribution: top RF importances present in form
                if self.train_result:
                    fi = get_rf_feature_importance(
                        self.train_result["models"]["Random Forest"],
                        self.train_result["numeric_features"],
                        self.train_result["categorical_features"], 8,
                    )
                    section_title(result_host, "Yếu tố đóng góp (Feature Importance — RF)")
                    show_dataframe(result_host, fi, height=160, page_size=8)
            except Exception as exc:  # noqa: BLE001
                messagebox.showerror("Dự báo", str(exc))

        primary_button(root, "Chấm điểm rủi ro", run, width=200, height=42).pack(anchor="w", padx=10, pady=8)

    def _load_model(self):
        if self.train_result and self.eval_result:
            best = self.eval_result["best_model_name"]
            return self.train_result["models"][best], self.train_result["feature_columns"], best
        meta_path = MODELS_DIR / "feature_meta.joblib"
        lr = MODELS_DIR / "logistic_regression.joblib"
        rf = MODELS_DIR / "random_forest.joblib"
        if meta_path.exists() and (lr.exists() or rf.exists()):
            meta = joblib.load(meta_path)
            # nếu target/dataset đổi → retrain
            if meta.get("target") != self.target:
                raise ValueError("Model không khớp target hiện tại — huấn luyện lại.")
            best = meta.get("best_model_name", "Random Forest")
            path = lr if best == "Logistic Regression" and lr.exists() else rf
            return joblib.load(path), meta["feature_columns"], best
        tr = train_classification_models(self.df_raw, save=True, target=self.target)
        ev = evaluate_all_classifiers(tr["models"], tr["X_test"], tr["y_test"])
        self.train_result, self.eval_result = tr, ev
        self.model_status = ev["best_model_name"]
        return tr["models"][ev["best_model_name"]], tr["feature_columns"], ev["best_model_name"]

    # ============================================================ INSIGHTS
    def _page_insights(self) -> None:
        root = self.content
        df = self._filtered()
        self._filter_caption(df)
        if df.empty:
            body_text(root, "Không còn bản ghi.")
            return
        insights = self._cached_insights(df)
        section_title(root, f"Insights ({len(insights)})")
        for i, ins in enumerate(insights, 1):
            insight_card(
                root, i, ins["group"], ins["title"], ins["insight"],
                evidence=ins.get("evidence"), difference=ins.get("difference"),
                severity=ins.get("severity"),
            )

    # ============================================================ RECS
    def _page_recs(self) -> None:
        root = self.content
        df = self._filtered()
        self._filter_caption(df)
        if df.empty:
            body_text(root, "Không còn bản ghi.")
            return
        insights = self._cached_insights(df)
        recs = generate_recommendations_dynamic(insights)
        section_title(root, f"Khuyến nghị ({len(recs)})")
        if not recs:
            body_text(root, "Không có khuyến nghị.", muted=True)
            return
        for i, r in enumerate(recs, 1):
            insight_card(
                root, i, r["priority"], r["based_on"],
                f"Problem: {r['problem']}\nEvidence: {r['evidence']}\nGoal: {r['expected_goal']}",
                recommendation=r["recommended_action"],
                severity=r["priority"],
            )


def main() -> None:
    app = PeopleRiskApp()
    app.mainloop()


if __name__ == "__main__":
    main()
