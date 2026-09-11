"""PeopleRisk AI — Desktop HR Analytics (dataset-driven)."""

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

from giao_dien.ve_bieu_do import (
    chart_attrition_donut,
    chart_boxplot_by_target,
    chart_rate_by_category,
    gallery_specs,
)
from giao_dien.toi_uu import AnalysisCache, debounce, df_content_id, filter_signature, tune_matplotlib_fast
from giao_dien.giao_dien_mau import FILTER_ROLE_LABELS, NAV_SECTIONS, THEME
from giao_dien.thanh_phan import (
    ScrollableFrame,
    body_text,
    clear_frame,
    close_all_dropdowns,
    configure_treeview_style,
    embed_figure,
    font,
    insight_card,
    make_kpi_card,
    option_menu,
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
    text_entry,
)
from xu_ly.thong_ke_mo_ta import descriptive_statistics
from xu_ly.phan_tich_eda import compute_kpis_dynamic, run_research_questions
from xu_ly.trang_thai_dataset import DatasetState, DatasetStatus
from xu_ly.ho_so_du_lieu import format_file_size, profile_dataset, suggest_filter_columns
from xu_ly.doc_du_lieu import COMMON_ENCODINGS, file_size_bytes, try_load_csv
from xu_ly.diem_chat_luong import calculate_quality_score
from xu_ly.tim_cot_du_lieu import build_schema, detect_target_candidates
from ket_qua.tao_insight import generate_insights_dynamic
from ket_qua.tao_khuyen_nghi import generate_recommendations_dynamic
from ket_qua.xuat_pdf import export_report_pdf
from mo_hinh.danh_gia import (
    evaluate_all_classifiers,
    evaluate_regression,
    get_rf_feature_importance,
    plot_confusion_matrix,
    plot_feature_importance,
    plot_roc_curves,
)
from mo_hinh.du_doan import predict_attrition
from mo_hinh.huan_luyen import train_classification_models, train_salary_regression
from xu_ly.lam_sach import clean_dataset, detect_outliers_iqr
from xu_ly.chuan_bi_du_lieu import get_feature_columns
from xu_ly.tien_ich import DEFAULT_DATASET_PATH, MODELS_DIR, apply_filters, format_vnd, format_vnd_compact

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

        self.state = DatasetState()

        # Pending upload (chưa confirm)
        self.pending_df: pd.DataFrame | None = None
        self.pending_name: str | None = None
        self.pending_path: str | None = None
        self.pending_encoding: str = "utf-8"
        self.pending_size: int | None = None
        self.pending_profile: dict | None = None
        self.pending_schema: dict | None = None
        self.pending_target_var: ctk.StringVar | None = None
        self.encoding_var = ctk.StringVar(value="utf-8")
        self.upload_status_lbl: ctk.CTkLabel | None = None

        self.predict_vars: dict[str, Any] = {}
        self.predict_step = 0
        self.filter_vars: dict[str, Any] = {}
        self.current_page = "upload"
        self._nav_lookup = {k: (lb, h) for _, items in NAV_SECTIONS for k, lb, h in items}
        self.sidebar_dataset_lbl = None
        self.cache = AnalysisCache()
        self._dataset_id = "empty"
        self._page_gen = 0
        self._clean_report: dict | None = None

        configure_treeview_style(self)
        self._build_shell()
        self.bind_all("<Escape>", lambda _e: close_all_dropdowns())
        if not self._load_default_dataset():
            self.show_page("upload")

    # ============================================================ PROPERTIES
    @property
    def df(self) -> pd.DataFrame:
        return self.state.active_df

    @property
    def schema(self) -> dict:
        return self.state.schema or {}

    @property
    def target(self) -> str | None:
        return self.state.target

    @property
    def has_dataset(self) -> bool:
        return self.state.ready

    @property
    def filter_state(self) -> dict[str, str]:
        return self.state.filter_state

    @property
    def train_result(self):
        return self.state.train_result

    @train_result.setter
    def train_result(self, value) -> None:
        self.state.train_result = value

    @property
    def eval_result(self):
        return self.state.eval_result

    @eval_result.setter
    def eval_result(self, value) -> None:
        self.state.eval_result = value

    @property
    def salary_eval(self):
        return self.state.salary_eval

    @salary_eval.setter
    def salary_eval(self, value) -> None:
        self.state.salary_eval = value

    @property
    def model_status(self) -> str:
        return self.state.model_status

    @model_status.setter
    def model_status(self, value: str) -> None:
        self.state.model_status = value

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
        secondary_button(right, "Xuất PDF", self._export_pdf, width=100, height=30).pack(side="left", padx=3)
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
        if not self.sidebar_dataset_lbl:
            return
        if not self.has_dataset:
            self.sidebar_dataset_lbl.configure(text="No dataset loaded")
            return
        n, m = self.df.shape
        name = self.state.name or "dataset"
        tgt = self.target or "—"
        self.sidebar_dataset_lbl.configure(text=f"{name}\n{n:,} × {m}\nTarget: {tgt}")

    def _set_header(self, key: str) -> None:
        label, hint = self._nav_lookup.get(key, ("", ""))
        self.header_label.configure(text=label)
        self.subheader_label.configure(text=hint)

        if self.has_dataset and self.state.loaded_at:
            loaded = self.state.loaded_at.strftime("%d/%m/%Y %H:%M")
            n, m = self.df.shape
            model = self.eval_result["best_model_name"] if self.eval_result else self.model_status
            self.status_label.configure(text=f"PeopleRisk AI  ·  {loaded}")
            self.footer_right.configure(
                text=f"{self.state.name}  ·  {n:,}×{m}  ·  Target:{self.target}  ·  Model:{model}"
            )
            clear_frame(self.model_pill_host)
            kind = "success" if self.eval_result else "warning"
            status_pill(self.model_pill_host, f"Model: {model}", kind).pack()
        else:
            st = self.state.status.value if self.state.status else "empty"
            self.status_label.configure(text=f"PeopleRisk AI  ·  status: {st}")
            self.footer_right.configure(text="No dataset loaded")
            clear_frame(self.model_pill_host)
            status_pill(self.model_pill_host, "No dataset", "warning").pack()

    def _highlight(self, key: str) -> None:
        for k, btn in self.nav_buttons.items():
            if k == key:
                btn.configure(fg_color=THEME.accent, text_color="#FFF", font=font(12, "bold"))
            else:
                btn.configure(fg_color="transparent", text_color=THEME.sidebar_text, font=font(12))

    def show_page(self, key: str) -> None:
        close_all_dropdowns()
        self.current_page = key
        self._page_gen += 1
        self._highlight(key)
        self._set_header(key)
        clear_frame(self.content)
        plt.close("all")

        if key != "upload" and not self.has_dataset:
            self._page_need_dataset()
            return

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

    def _page_need_dataset(self) -> None:
        root = self.content
        box = panel(root, "Dataset required", "No dataset loaded")
        box.pack(fill="x", padx=12, pady=40)
        body_text(box, "Upload a CSV dataset to start HR Analytics.")
        primary_button(
            box, "Go to Dataset Import", lambda: self.show_page("upload"), width=220
        ).pack(anchor="w", padx=12, pady=14)

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
        roles = self.schema.get("roles") or {}
        return self.cache.set(key, generate_insights_dynamic(df, roles, self.target))

    def _cached_kpis(self, df: pd.DataFrame):
        key = self.cache.key("kpi", self._dataset_id, self.target, len(df), self._filter_sig())
        hit = self.cache.get(key)
        if hit is not None:
            return hit
        return self.cache.set(key, compute_kpis_dynamic(df, self.schema.get("roles") or {}, self.target))

    def _cached_rq(self, df: pd.DataFrame):
        key = self.cache.key("rq", self._dataset_id, self.target, len(df), self._filter_sig())
        hit = self.cache.get(key)
        if hit is not None:
            return hit
        return self.cache.set(key, run_research_questions(df, self.schema.get("roles") or {}, self.target))

    def _refresh(self) -> None:
        self.show_page(self.current_page)

    def _clear_analysis_cache(self) -> None:
        self.cache.clear()
        self.state.cache_store.clear()

    def _clear_pending(self) -> None:
        self.pending_df = None
        self.pending_name = None
        self.pending_path = None
        self.pending_encoding = "utf-8"
        self.pending_size = None
        self.pending_profile = None
        self.pending_schema = None
        self.pending_target_var = None
        self._clean_report = None

    # ============================================================ FILTERS
    def _draw_filters(self, on_change) -> None:
        if not self.has_dataset:
            return
        cols = [c for c in suggest_filter_columns(self.df, self.schema) if c in self.df.columns]
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
                {str(x) for x in self.df[col].dropna().tolist()}, key=str
            )
            current = str(saved.get(col, "Tất cả"))
            if current not in opts:
                current = "Tất cả"
            var = ctk.StringVar(value=current)
            self.filter_vars[col] = var
            self.filter_state[col] = current

            menu = option_menu(
                cell, values=opts, variable=var, width=140, height=28, font=font(11),
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
        self.after(50, lambda: setattr(self, "_filters_ready", True))

    def _clear_filters(self) -> None:
        self.filter_state.clear()
        self.filter_vars.clear()
        self.show_page(self.current_page)

    def _active_filters(self) -> dict[str, str]:
        if not self.has_dataset:
            return {}
        return {
            c: str(v)
            for c, v in self.filter_state.items()
            if str(v).strip() and str(v) != "Tất cả" and c in self.df.columns
        }

    def _filtered(self) -> pd.DataFrame:
        return apply_filters(self.df, self._active_filters())

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
            f"Đang phân tích {len(df):,} / {len(self.df):,} nhân viên  ·  {detail}  ·  {self.state.name}",
            muted=True,
        )

    def _report_payload(self) -> dict[str, Any]:
        """Gói dữ liệu báo cáo theo bộ lọc hiện tại (Excel + PDF dùng chung)."""
        df = self._filtered()
        roles = self.schema.get("roles") or {}
        kpis = compute_kpis_dynamic(df, roles, self.target)
        insights = generate_insights_dynamic(df, roles, self.target)
        recs = generate_recommendations_dynamic(insights)
        filters = self._active_filters()
        if filters:
            filter_note = ", ".join(f"{k}={v}" for k, v in filters.items())
        else:
            filter_note = "Không lọc"
        model_name = None
        model_metrics = None
        metrics_table = None
        if self.eval_result is not None:
            model_name = self.eval_result.get("best_model_name")
            metrics_table = self.eval_result.get("metrics_table")
            if model_name and model_name in (self.eval_result.get("results") or {}):
                model_metrics = self.eval_result["results"][model_name].get("metrics")
        return {
            "df": df,
            "kpis": kpis,
            "insights": insights,
            "recs": recs,
            "filter_note": filter_note,
            "model_name": model_name,
            "model_metrics": model_metrics,
            "metrics_table": metrics_table,
        }

    def _export(self) -> None:
        if not self.has_dataset:
            messagebox.showwarning("Xuất báo cáo", "Chưa có dataset.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".xlsx", filetypes=[("Excel", "*.xlsx")],
            initialfile=f"PeopleRisk_{datetime.now():%Y%m%d_%H%M}.xlsx",
        )
        if not path:
            return
        try:
            payload = self._report_payload()
            with pd.ExcelWriter(path, engine="openpyxl") as w:
                pd.DataFrame([payload["kpis"]]).to_excel(w, sheet_name="KPI", index=False)
                pd.DataFrame(payload["insights"]).to_excel(w, sheet_name="Insights", index=False)
                pd.DataFrame(payload["recs"]).to_excel(w, sheet_name="Recommendations", index=False)
                if payload["metrics_table"] is not None:
                    payload["metrics_table"].to_excel(w, sheet_name="Model", index=False)
            messagebox.showinfo("Xuất Excel", f"Đã lưu:\n{path}")
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Lỗi Excel", str(exc))

    def _export_pdf(self) -> None:
        if not self.has_dataset:
            messagebox.showwarning("Xuất PDF", "Chưa có dataset.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".pdf",
            filetypes=[("PDF", "*.pdf")],
            initialfile=f"PeopleRisk_{datetime.now():%Y%m%d_%H%M}.pdf",
        )
        if not path:
            return
        try:
            payload = self._report_payload()
            df = payload["df"]
            export_report_pdf(
                path,
                dataset_name=self.state.name,
                target=self.target,
                n_rows=len(df),
                n_cols=int(df.shape[1]) if not df.empty else 0,
                filter_note=payload["filter_note"],
                kpis=payload["kpis"],
                insights=payload["insights"],
                recommendations=payload["recs"],
                model_name=payload["model_name"],
                model_metrics=payload["model_metrics"],
                metrics_table=payload["metrics_table"],
            )
            messagebox.showinfo("Xuất PDF", f"Đã lưu:\n{path}")
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Lỗi PDF", str(exc))

    # ============================================================ UPLOAD
    def _page_upload(self) -> None:
        root = self.content
        status = self.state.status.value
        if self.pending_df is not None:
            status = "loading" if self.state.status == DatasetStatus.LOADING else "preview"
        elif self.has_dataset:
            status = "ready"

        box = panel(root, "Nhập Dataset", f"Status: {status}")
        box.pack(fill="x", padx=8, pady=10)

        enc_row = ctk.CTkFrame(box, fg_color="transparent")
        enc_row.pack(fill="x", padx=12, pady=(8, 4))
        ctk.CTkLabel(enc_row, text="Encoding", font=font(11, "bold"), text_color=THEME.text_muted).pack(
            side="left", padx=(0, 8)
        )
        option_menu(
            enc_row,
            values=list(COMMON_ENCODINGS),
            variable=self.encoding_var,
            width=140,
            height=30,
        ).pack(side="left")

        btns = ctk.CTkFrame(box, fg_color="transparent")
        btns.pack(padx=12, pady=8, anchor="w")
        primary_button(btns, "Chọn file CSV", self._pick_csv, width=160).pack(side="left", padx=4)
        if self.has_dataset and self.state.path:
            secondary_button(btns, "Reload Dataset", self._reload_dataset, width=150).pack(side="left", padx=4)
        if self.has_dataset or self.pending_df is not None:
            secondary_button(btns, "Remove Dataset", self._remove_dataset, width=150).pack(side="left", padx=4)

        self.upload_status_lbl = ctk.CTkLabel(
            box, text="", font=font(11), text_color=THEME.text_muted, justify="left"
        )
        self.upload_status_lbl.pack(anchor="w", padx=12, pady=(0, 8))

        if self.state.status == DatasetStatus.ERROR and self.state.error_message:
            body_text(box, f"Error: {self.state.error_message}", muted=False)

        if self.pending_df is None and not self.has_dataset:
            body_text(
                root,
                "No dataset loaded. Chọn file CSV để bắt đầu phân tích HR.",
                muted=True,
            )
            return

        # Hiển thị preview từ pending, hoặc dataset đang active
        if self.pending_df is not None:
            df = self.pending_df
            name = self.pending_name or "upload.csv"
            size = self.pending_size
            profile = self.pending_profile or profile_dataset(df)
            encoding = self.pending_encoding
            is_pending = True
        else:
            df = self.state.df if self.state.df is not None else self.df
            name = self.state.name or "dataset.csv"
            size = self.state.file_size_bytes
            profile = self.state.profile or profile_dataset(df, target=self.target)
            encoding = self.state.encoding
            is_pending = False

        section_title(root, "Dataset preview")
        info = ctk.CTkFrame(root, fg_color="transparent")
        info.pack(fill="x", padx=4)
        for i in range(4):
            info.grid_columnconfigure(i, weight=1)
        make_kpi_card(info, "File", name, tone="brand").grid(row=0, column=0, sticky="nsew", padx=3)
        make_kpi_card(info, "Size", format_file_size(size), tone="accent").grid(row=0, column=1, sticky="nsew", padx=3)
        make_kpi_card(info, "Rows", f"{profile.get('row_count', len(df)):,}", tone="info").grid(
            row=0, column=2, sticky="nsew", padx=3
        )
        make_kpi_card(info, "Columns", f"{profile.get('column_count', df.shape[1])}", tone="warning").grid(
            row=0, column=3, sticky="nsew", padx=3
        )
        body_text(root, f"Encoding: {encoding}", muted=True)

        section_title(root, "Columns & dtypes")
        cols_meta = profile.get("columns") or []
        if cols_meta:
            show_dataframe(root, pd.DataFrame(cols_meta), height=180, page_size=10, enable_search=True)
        else:
            show_dataframe(
                root,
                pd.DataFrame({"Column": df.columns, "Dtype": [str(df[c].dtype) for c in df.columns]}),
                height=160,
                page_size=10,
            )

        candidates = profile.get("target_candidates") or detect_target_candidates(df)
        suggested = ", ".join(candidates[:5]) if candidates else "—"
        section_title(root, "Chọn biến mục tiêu (Target)")
        body_text(root, f"Suggested targets: {suggested}", muted=True)
        all_cols = list(df.columns)
        default_tgt = (
            (self.target if self.target in all_cols else None)
            or (candidates[0] if candidates else None)
            or (all_cols[0] if all_cols else "")
        )
        self.pending_target_var = ctk.StringVar(value=str(default_tgt))
        option_menu(
            root, values=all_cols or [""], variable=self.pending_target_var, width=280, height=34
        ).pack(anchor="w", padx=12, pady=6)

        section_title(root, "Preview (15 rows)")
        show_dataframe(root, df.head(15), height=200, page_size=10)

        if is_pending:
            primary_button(root, "Confirm Dataset", self._confirm_pending, width=200).pack(
                anchor="w", padx=12, pady=14
            )
        elif self.has_dataset:
            body_text(root, f"Dataset sẵn sàng · Target: {self.target}", muted=True)
            primary_button(root, "Confirm Dataset", self._confirm_pending, width=200).pack(
                anchor="w", padx=12, pady=14
            )

    def _pick_csv(self) -> None:
        path = filedialog.askopenfilename(filetypes=[("CSV", "*.csv"), ("All", "*.*")])
        if not path:
            return
        self.state.status = DatasetStatus.LOADING
        self.state.error_message = ""
        encoding = self.encoding_var.get() or "utf-8"
        if self.upload_status_lbl:
            self.upload_status_lbl.configure(text="Loading…")
        self.update_idletasks()
        try:
            df, used_enc = try_load_csv(path, encoding=encoding)
            size = file_size_bytes(path)
            profile = profile_dataset(df)
            schema = build_schema(df)
            self.pending_df = df
            self.pending_name = Path(path).name
            self.pending_path = path
            self.pending_encoding = used_enc
            self.pending_size = size
            self.pending_profile = profile
            self.pending_schema = schema
            self.encoding_var.set(used_enc)
            self.state.status = DatasetStatus.EMPTY  # pending until confirm
            self.show_page("upload")
        except Exception as exc:  # noqa: BLE001
            self.state.status = DatasetStatus.ERROR
            self.state.error_message = str(exc)
            self._clear_pending()
            messagebox.showerror("Lỗi CSV", str(exc))
            self.show_page("upload")

    def _confirm_pending(self) -> None:
        # Confirm từ pending hoặc re-confirm target trên dataset hiện tại
        if self.pending_df is not None:
            df = self.pending_df
            name = self.pending_name or "upload.csv"
            path = self.pending_path
            encoding = self.pending_encoding
            size = self.pending_size
        elif self.has_dataset and self.state.df is not None:
            df = self.state.df
            name = self.state.name or "dataset.csv"
            path = self.state.path
            encoding = self.state.encoding
            size = self.state.file_size_bytes
        else:
            messagebox.showwarning("Dataset", "Chưa có file để xác nhận.")
            return

        if self.pending_target_var is None:
            return
        target = self.pending_target_var.get()
        if not target or target not in df.columns:
            messagebox.showerror("Target", f"Cột target `{target}` không tồn tại.")
            return

        schema = build_schema(df, target=target)
        profile = profile_dataset(df, target=target)
        self.state.activate(
            df,
            name=name,
            path=path,
            encoding=encoding,
            file_size_bytes=size,
            schema=schema,
            target=target,
            profile=profile,
        )
        self._dataset_id = df_content_id(df)
        self._clear_analysis_cache()
        self._clear_pending()
        self.filter_vars.clear()
        self._refresh_sidebar_dataset()
        messagebox.showinfo("Dataset", f"Đã kích hoạt: {name}\nTarget: {target}")
        self.show_page("dashboard")

    def _reload_dataset(self) -> None:
        if not self.state.path:
            messagebox.showwarning("Reload", "Không có đường dẫn file để đọc lại.")
            return
        path = self.state.path
        encoding = self.encoding_var.get() or self.state.encoding or "utf-8"
        self.state.status = DatasetStatus.LOADING
        self.update_idletasks()
        try:
            df, used_enc = try_load_csv(path, encoding=encoding)
            size = file_size_bytes(path)
            target = self.target if self.target in df.columns else None
            if target is None:
                cands = detect_target_candidates(df)
                target = cands[0] if cands else (list(df.columns)[0] if len(df.columns) else None)
            schema = build_schema(df, target=target)
            profile = profile_dataset(df, target=target)
            self.state.activate(
                df,
                name=Path(path).name,
                path=path,
                encoding=used_enc,
                file_size_bytes=size,
                schema=schema,
                target=target,
                profile=profile,
            )
            self._dataset_id = df_content_id(df)
            self._clear_analysis_cache()
            self._clear_pending()
            self.filter_vars.clear()
            self.encoding_var.set(used_enc)
            self._refresh_sidebar_dataset()
            messagebox.showinfo("Reload", f"Đã đọc lại: {Path(path).name}")
            self.show_page("upload")
        except Exception as exc:  # noqa: BLE001
            self.state.status = DatasetStatus.ERROR
            self.state.error_message = str(exc)
            messagebox.showerror("Reload", str(exc))
            self.show_page("upload")

    def _remove_dataset(self) -> None:
        self.state.clear()
        self._clear_analysis_cache()
        self._clear_pending()
        self.filter_vars.clear()
        self._dataset_id = "empty"
        self._refresh_sidebar_dataset()
        self.show_page("upload")

    def _load_default_dataset(self) -> bool:
        """Nạp CSV mặc định trong data/raw nếu có. Vẫn đổi dataset được qua Nhập Dataset."""
        path = DEFAULT_DATASET_PATH
        if not path.is_file():
            return False
        try:
            df, used_enc = try_load_csv(str(path), encoding="utf-8")
            size = file_size_bytes(path)
            cands = detect_target_candidates(df)
            target = cands[0] if cands else (list(df.columns)[0] if len(df.columns) else None)
            if not target:
                return False
            schema = build_schema(df, target=target)
            profile = profile_dataset(df, target=target)
            self.state.activate(
                df,
                name=path.name,
                path=str(path),
                encoding=used_enc,
                file_size_bytes=size,
                schema=schema,
                target=target,
                profile=profile,
            )
            self._dataset_id = df_content_id(df)
            self._clear_analysis_cache()
            self._clear_pending()
            self.filter_vars.clear()
            self.encoding_var.set(used_enc)
            self._refresh_sidebar_dataset()
            self.show_page("dashboard")
            return True
        except Exception:  # noqa: BLE001
            return False

    # ============================================================ DASHBOARD
    def _page_dashboard(self) -> None:
        root = self.content
        self._draw_filters(lambda: self.show_page("dashboard"))
        df = self._filtered()
        self._filter_caption(df)
        if df.empty:
            body_text(root, "Không còn bản ghi sau khi lọc.")
            return
        roles = self.schema.get("roles") or {}
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
        if roles.get("income") and kpis.get("avg_income") is not None:
            extras.append(("Thu nhập TB", format_vnd_compact(kpis["avg_income"]), format_vnd(kpis["avg_income"]) + "/tháng", "accent"))
        elif not roles.get("income"):
            extras.append(("Thu nhập TB", "N/A", "no salary column", "warning"))
        if kpis.get("avg_years_company") is not None:
            extras.append(("Thâm niên TB", f"{kpis['avg_years_company']:.2f} năm", "tại công ty", "brand"))
        if kpis.get("avg_job_satisfaction") is not None:
            extras.append(("Hài lòng CV", f"{kpis['avg_job_satisfaction']:.2f}", "thang 1–5", "success"))
        for i in range(max(len(extras), 1)):
            row2.grid_columnconfigure(i, weight=1)
        for i, (t, v, s, tone) in enumerate(extras):
            make_kpi_card(row2, t, v, s, tone).grid(row=0, column=i, sticky="nsew", padx=3)

        if not roles.get("income"):
            body_text(root, "Salary analysis unavailable — no salary column detected.", muted=True)

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
        tgt = self.target

        def _draw_left() -> None:
            loading_l.destroy()
            if not tgt or tgt not in df.columns:
                body_text(L, "Không có cột target để vẽ attrition.", muted=True)
                return
            fig, note, err = safe_chart(lambda: chart_attrition_donut(df, tgt))
            embed_figure(L, fig, 240, "Active vs Left", note, err)
            if fig:
                plt.close(fig)

        def _draw_right() -> None:
            loading_r.destroy()
            if dept and tgt:
                fig, note, err = safe_chart(
                    lambda: chart_rate_by_category(df, dept, tgt, "Attrition by Department")
                )
                embed_figure(R, fig, 240, "Theo phòng ban", note, err)
                if fig:
                    plt.close(fig)
            else:
                body_text(R, "Không có cột phòng ban trong dataset.", muted=True)

        self._defer(10, _draw_left)
        self._defer(60, _draw_right)

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
        df = self.df
        profile = self.state.profile or profile_dataset(df, target=self.target)
        sch = self.schema

        row = ctk.CTkFrame(root, fg_color="transparent")
        row.pack(fill="x", padx=4, pady=6)
        for i in range(4):
            row.grid_columnconfigure(i, weight=1)
        make_kpi_card(row, "Records", f"{profile.get('row_count', len(df)):,}", tone="brand").grid(
            row=0, column=0, sticky="nsew", padx=3
        )
        make_kpi_card(row, "Variables", f"{profile.get('column_count', df.shape[1])}", tone="accent").grid(
            row=0, column=1, sticky="nsew", padx=3
        )
        make_kpi_card(row, "Numeric", f"{profile.get('numeric_count', sch.get('n_numeric', 0))}", tone="info").grid(
            row=0, column=2, sticky="nsew", padx=3
        )
        make_kpi_card(
            row, "Categorical", f"{profile.get('categorical_count', sch.get('n_categorical', 0))}", tone="warning"
        ).grid(row=0, column=3, sticky="nsew", padx=3)

        row2 = ctk.CTkFrame(root, fg_color="transparent")
        row2.pack(fill="x", padx=4, pady=2)
        for i in range(4):
            row2.grid_columnconfigure(i, weight=1)
        miss = profile.get("missing_cells", sch.get("missing_total", 0))
        dup = profile.get("duplicate_rows", sch.get("duplicate_total", 0))
        make_kpi_card(
            row2, "Missing", str(miss), "Good" if miss == 0 else "Check",
            "success" if miss == 0 else "warning",
        ).grid(row=0, column=0, sticky="nsew", padx=3)
        make_kpi_card(
            row2, "Duplicate", str(dup), "Good" if dup == 0 else "Check",
            "success" if dup == 0 else "warning",
        ).grid(row=0, column=1, sticky="nsew", padx=3)
        make_kpi_card(row2, "Target", str(self.target or "—"), "Confirmed", "accent").grid(
            row=0, column=2, sticky="nsew", padx=3
        )
        make_kpi_card(
            row2, "Datetime", str(profile.get("datetime_count", 0)), "profile", "info"
        ).grid(row=0, column=3, sticky="nsew", padx=3)

        section_title(root, "Data Dictionary")
        cols_meta = profile.get("columns")
        if cols_meta:
            rows = []
            for meta in cols_meta:
                col = meta["Column"]
                role = "Target" if col == self.target else ("ID" if col == sch.get("id_col") else "Feature")
                rows.append({**meta, "Role": role})
            show_dataframe(root, pd.DataFrame(rows), height=240, page_size=12, enable_search=True)
        else:
            rows = []
            for col in df.columns:
                role = "Target" if col == self.target else ("ID" if col == sch.get("id_col") else "Feature")
                rows.append({
                    "Column": col,
                    "Data Type": str(df[col].dtype),
                    "Non-null": int(df[col].notna().sum()),
                    "Null": int(df[col].isna().sum()),
                    "Unique": int(df[col].nunique()),
                    "Role": role,
                })
            show_dataframe(root, pd.DataFrame(rows), height=240, page_size=12, enable_search=True)

        section_title(root, "Sample Data (20 dòng)")
        show_dataframe(root, df.head(20), height=220, page_size=10, enable_search=True)

    # ============================================================ QUALITY
    def _page_quality(self) -> None:
        root = self.content
        df = self.df
        id_col = self.schema.get("id_col")
        report = calculate_quality_score(df, id_col=id_col)
        self.state.quality_report = report

        row = ctk.CTkFrame(root, fg_color="transparent")
        row.pack(fill="x", padx=6, pady=10)
        for i, (label, key) in enumerate([
            ("Completeness", "completeness"),
            ("Consistency", "consistency"),
            ("Validity", "validity"),
            ("Uniqueness", "uniqueness"),
        ]):
            row.grid_columnconfigure(i, weight=1)
            make_kpi_card(row, label, f"{report[key]:.1f}%", report["label"], "brand").grid(
                row=0, column=i, sticky="nsew", padx=3
            )

        score = report["score"]
        qcard = ctk.CTkFrame(
            root,
            fg_color=THEME.success_soft if score >= 90 else THEME.warning_soft,
            corner_radius=10,
            height=72,
        )
        qcard.pack(fill="x", padx=8, pady=6)
        ctk.CTkLabel(
            qcard,
            text=f"Quality Score: {score}% — {report['label']}  ·  {report.get('message', '')}",
            font=font(13, "bold"),
            text_color=THEME.success if score >= 90 else THEME.warning,
        ).pack(padx=12, pady=16, anchor="w")

        status_row = ctk.CTkFrame(root, fg_color="transparent")
        status_row.pack(fill="x", padx=6, pady=4)
        quality_status_card(status_row, "Missing", report.get("missing", 0), "0 Missing").pack(side="left", padx=6)
        quality_status_card(status_row, "Duplicate", report.get("duplicates", 0), "0 Duplicate").pack(side="left", padx=6)

        section_title(root, "Issues")
        issues = report.get("issues") or []
        if not issues:
            body_text(root, "Không phát hiện vấn đề chất lượng đáng kể.", muted=True)
        else:
            for iss in issues:
                body_text(root, f"• {iss}")

        section_title(root, "Outlier IQR (không tự xóa)")
        outliers = report.get("outliers")
        if outliers is None or (isinstance(outliers, pd.DataFrame) and outliers.empty):
            outliers = detect_outliers_iqr(df)
        show_dataframe(root, outliers, height=160, page_size=8)

        if self._clean_report:
            section_title(root, "Clean result (before → after)")
            cr = self._clean_report
            body_text(
                root,
                f"Rows {cr.get('rows_before')} → {cr.get('rows_after')}  ·  "
                f"Cols {cr.get('cols_before')} → {cr.get('cols_after', cr.get('cols_before'))}  ·  "
                f"Missing {cr.get('missing_before')} → {cr.get('missing_after')}  ·  "
                f"Dups {cr.get('duplicates_before')} → {cr.get('duplicates_after')}",
            )

        primary_button(root, "Clean dataset", self._run_clean, width=180).pack(anchor="w", padx=10, pady=10)

    def _run_clean(self) -> None:
        if not self.has_dataset or self.state.df is None:
            return
        try:
            cleaned, report = clean_dataset(self.state.df, drop_duplicates=True, save=False)
            self.state.df_cleaned = cleaned
            self._clean_report = report
            # cập nhật profile theo active df
            self.state.profile = profile_dataset(self.df, target=self.target)
            self._dataset_id = df_content_id(self.df)
            self._clear_analysis_cache()
            messagebox.showinfo(
                "Clean",
                f"Rows {report['rows_before']}→{report['rows_after']}\n"
                f"Missing {report['missing_before']}→{report.get('missing_after')}\n"
                f"Dups {report['duplicates_before']}→{report.get('duplicates_after')}",
            )
            self.show_page("quality")
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Clean", str(exc))

    # ============================================================ EDA
    def _page_eda(self) -> None:
        root = self.content
        self._draw_filters(lambda: self.show_page("eda"))
        df = self._filtered()
        self._filter_caption(df)
        if df.empty:
            body_text(root, "Không còn bản ghi.")
            return
        roles = self.schema.get("roles") or {}
        rq = self._cached_rq(df)

        num_cols = [c for c in (self.schema.get("numeric") or []) if c in df.columns][:16]
        if num_cols:
            section_title(root, "Thống kê mô tả")
            show_dataframe(root, descriptive_statistics(df, num_cols), height=200, page_size=10)

        tgt = self.target
        chart_map = {
            "rq1": (lambda: chart_attrition_donut(df, tgt)) if tgt else None,
            "rq2": (lambda: chart_rate_by_category(df, roles["department"], tgt, "Attrition by Department"))
            if roles.get("department") and tgt else None,
            "rq3": (lambda: chart_rate_by_category(df, roles["overtime"], tgt, "Attrition by Overtime"))
            if roles.get("overtime") and tgt else None,
            "rq4": (lambda: chart_rate_by_category(df, roles["job_satisfaction"], tgt, "By Satisfaction"))
            if roles.get("job_satisfaction") and tgt else None,
            "rq5": (lambda: chart_boxplot_by_target(df, roles["income"], tgt, "Income", roles["income"]))
            if roles.get("income") and tgt else None,
            "rq6": (lambda: chart_boxplot_by_target(df, roles["tenure"], tgt, "Tenure", roles["tenure"]))
            if roles.get("tenure") and tgt else None,
            "rq7": (lambda: chart_boxplot_by_target(df, roles["distance"], tgt, "Distance", roles["distance"]))
            if roles.get("distance") and tgt else None,
        }
        if not roles.get("income"):
            body_text(root, "Salary analysis unavailable — no salary column detected.", muted=True)

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

        specs = gallery_specs(df, self.schema.get("roles") or {}, self.target)
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
        tgt = self.target
        intro = panel(root, "Phân loại nghỉ việc", f"Target: {tgt or '—'}  ·  Train/Test 80/20 (stratify)")
        intro.pack(fill="x", padx=6, pady=6)
        status = ctk.CTkLabel(intro, text="", font=font(11), text_color=THEME.text_muted)
        status.pack(anchor="w", padx=12)

        profile = self.state.profile or {}
        dt_count = profile.get("datetime_count", 0)
        if not dt_count:
            body_text(
                intro,
                "No datetime column detected. Time-series forecasting is unavailable for this dataset.",
                muted=True,
            )

        def _target_ok() -> str | None:
            if not tgt or tgt not in self.df.columns:
                return "No target column selected. Choose a target on the Dataset Import page."
            n_cls = int(self.df[tgt].nunique(dropna=True))
            if n_cls < 2:
                return (
                    f"Target `{tgt}` has only {n_cls} class. "
                    "Classification requires at least 2 classes."
                )
            return None

        def train_cls() -> None:
            err = _target_ok()
            if err:
                messagebox.showerror("Train error", err)
                return
            status.configure(text="Đang train Logistic Regression & Random Forest...")
            self.update_idletasks()
            try:
                tr = train_classification_models(self.df, save=True, target=tgt)
                ev = evaluate_all_classifiers(tr["models"], tr["X_test"], tr["y_test"])
                self.train_result, self.eval_result = tr, ev
                self.model_status = ev["best_model_name"]
                meta_path = MODELS_DIR / "feature_meta.joblib"
                meta = joblib.load(meta_path) if meta_path.exists() else {}
                meta.update({
                    "best_model_name": ev["best_model_name"],
                    "target": tgt,
                    "feature_columns": tr["feature_columns"],
                })
                joblib.dump(meta, meta_path)
                self.show_page("model")
            except Exception as exc:  # noqa: BLE001
                messagebox.showerror("Train error", str(exc))

        def train_reg() -> None:
            roles = self.schema.get("roles") or {}
            if not roles.get("income"):
                messagebox.showwarning(
                    "Regression",
                    "Salary analysis unavailable — no salary column detected.",
                )
                return
            status.configure(text="Đang train Salary Regression...")
            self.update_idletasks()
            try:
                sal = train_salary_regression(self.df, save=True)
                self.salary_eval = evaluate_regression(sal["pipeline"], sal["X_test"], sal["y_test"])
                self.show_page("model")
            except Exception as exc:  # noqa: BLE001
                messagebox.showwarning("Regression", str(exc))

        btns = ctk.CTkFrame(intro, fg_color="transparent")
        btns.pack(anchor="w", padx=10, pady=8)
        primary_button(btns, "Huấn luyện phân loại", train_cls, width=180).pack(side="left", padx=3)
        secondary_button(btns, "Huấn luyện hồi quy thu nhập", train_reg, width=210).pack(side="left", padx=3)

        pre_err = _target_ok()
        if pre_err and self.eval_result is None:
            body_text(root, pre_err, muted=True)

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
        roles = self.schema.get("roles") or {}
        if self.salary_eval is None:
            if not roles.get("income"):
                body_text(root, "Salary analysis unavailable — no salary column detected.", muted=True)
            else:
                body_text(root, "Chưa huấn luyện hồi quy thu nhập.", muted=True)
        else:
            m = self.salary_eval["metrics"]
            show_dataframe(root, pd.DataFrame([m]), height=70, page_size=3)
            body_text(root, f"MAE ≈ {format_vnd(m['MAE'])} · RMSE ≈ {format_vnd(m['RMSE'])} · R²={m['R2']:.4f}")

    # ============================================================ PREDICT
    def _current_feature_columns(self) -> tuple[list[str], list[str], list[str]]:
        """Feature sạch cho form/train: trừ target, ID, junk, cột hằng."""
        tgt = self.target
        numeric, categorical = get_feature_columns(self.df, target=tgt)
        feature_cols = list(numeric) + list(categorical)
        return feature_cols, numeric, categorical

    def _page_predict(self) -> None:
        root = self.content
        roles = self.schema.get("roles") or {}
        tgt = self.target

        if not tgt or tgt not in self.df.columns:
            body_text(root, "Chưa chọn target. Vào Nhập Dataset để chọn.")
            primary_button(root, "Go to Dataset Import", lambda: self.show_page("upload"), width=200).pack(
                anchor="w", padx=10, pady=10
            )
            return
        if int(self.df[tgt].nunique(dropna=True)) < 2:
            body_text(root, f"Target `{tgt}` chỉ có 1 lớp — không dự báo được.")
            return

        feature_cols, numeric, categorical = self._current_feature_columns()
        numeric_set = set(numeric)
        id_col = self.schema.get("id_col") or roles.get("id")
        junk_cols = [c for c in (self.schema.get("junk_cols") or []) if c and c != id_col]

        if not feature_cols:
            body_text(root, "Dataset không còn cột feature để nhập.")
            return

        trained = bool(self.train_result and self.eval_result)
        if not trained:
            warn = panel(root, "Chưa có mô hình", "Cần huấn luyện trước khi chấm điểm (hoặc bấm Chấm điểm để tự train).")
            warn.pack(fill="x", padx=6, pady=6)
            btns = ctk.CTkFrame(warn, fg_color="transparent")
            btns.pack(anchor="w", padx=10, pady=8)

            def _train_now() -> None:
                try:
                    self._ensure_classifier_trained(force=True)
                    self._set_header(self.current_page)
                    self.show_page("predict")
                except Exception as exc:  # noqa: BLE001
                    messagebox.showerror("Huấn luyện", str(exc))

            primary_button(btns, "Huấn luyện ngay", _train_now, width=160).pack(side="left", padx=3)
            secondary_button(btns, "Mở trang Mô hình AI", lambda: self.show_page("model"), width=180).pack(
                side="left", padx=3
            )
        else:
            body_text(
                root,
                f"Model: {self.eval_result['best_model_name']}  ·  Form {len(feature_cols)} feature"
                + (f"  ·  đã loại ID `{id_col}`" if id_col else "")
                + (f"  ·  junk: {', '.join(junk_cols)}" if junk_cols else "")
                + "  ·  mặc định = median/mode",
                muted=True,
            )

        # Nhóm theo role đã nhận diện; phần còn lại → "CÁC CỘT KHÁC"
        role_groups = [
            ("CÁ NHÂN", ["age", "gender", "marital", "location"]),
            ("CÔNG VIỆC", ["department", "job_role", "job_level", "contract", "travel", "education", "field"]),
            ("THU NHẬP", ["income", "salary_hike"]),
            ("LÀM VIỆC", ["overtime", "distance", "tenure", "total_experience", "num_companies",
                          "years_role", "years_promo", "years_manager"]),
            ("HÀI LÒNG", ["job_satisfaction", "env_satisfaction", "worklife", "involvement", "relationship"]),
            ("HIỆU SUẤT", ["performance", "training"]),
        ]
        assigned: set[str] = set()
        grouped: list[tuple[str, list[str]]] = []
        for title, role_keys in role_groups:
            cols_in = []
            for rk in role_keys:
                col = roles.get(rk)
                if col and col in feature_cols and col not in assigned:
                    cols_in.append(col)
                    assigned.add(col)
            if cols_in:
                grouped.append((title, cols_in))
        rest = [c for c in feature_cols if c not in assigned]
        if rest:
            grouped.append(("CÁC CỘT KHÁC", rest))

        self.predict_vars = {}

        def _default_var(col: str) -> tuple[str, ctk.StringVar]:
            if col in numeric_set or pd.api.types.is_numeric_dtype(self.df[col]):
                series = pd.to_numeric(self.df[col], errors="coerce")
                med = series.median()
                if pd.notna(med):
                    val = str(int(round(float(med)))) if float(med).is_integer() else str(round(float(med), 4))
                else:
                    val = "0"
                return "num", ctk.StringVar(value=val)
            opts = sorted(self.df[col].dropna().astype(str).unique().tolist())
            mode = self.df[col].mode()
            default = str(mode.iloc[0]) if len(mode) else (opts[0] if opts else "")
            return "cat", ctk.StringVar(value=default)

        def _add_field(parent, col: str) -> None:
            if col in self.predict_vars:
                return
            kind, var = _default_var(col)
            self.predict_vars[col] = (kind, var)
            cell = ctk.CTkFrame(parent, fg_color="transparent")
            cell.pack(fill="x", padx=10, pady=3)
            ctk.CTkLabel(cell, text=col, font=font(10, "bold"), text_color=THEME.text_muted).pack(anchor="w")
            if kind == "num":
                text_entry(cell, textvariable=var, height=28).pack(fill="x")
            else:
                opts = sorted(self.df[col].dropna().astype(str).unique().tolist())
                if 0 < len(opts) <= 80:
                    option_menu(cell, values=opts, variable=var, height=28).pack(fill="x")
                else:
                    text_entry(cell, textvariable=var, height=28).pack(fill="x")

        form = ctk.CTkFrame(root, fg_color="transparent")
        form.pack(fill="x", padx=4, pady=4)
        n_cols = 3
        for i in range(n_cols):
            form.grid_columnconfigure(i, weight=1)
        cols_ui = [ctk.CTkFrame(form, fg_color="transparent") for _ in range(n_cols)]
        for i, c in enumerate(cols_ui):
            c.grid(row=0, column=i, sticky="nsew", padx=3)

        for gi, (title, cols_in) in enumerate(grouped):
            host = cols_ui[gi % n_cols]
            card = panel(host, f"{title} ({len(cols_in)})")
            card.pack(fill="x", pady=4)
            for col in cols_in:
                _add_field(card, col)

        for col in feature_cols:
            if col not in self.predict_vars:
                _add_field(cols_ui[0], col)

        section_title(root, f"Đã sẵn sàng {len(self.predict_vars)} / {len(feature_cols)} cột feature")

        result_host = ctk.CTkFrame(root, fg_color="transparent")
        result_host.pack(fill="x", padx=4, pady=8)

        def run() -> None:
            try:
                model, feats, name = self._ensure_classifier_trained(force=False)
                self._set_header(self.current_page)
                inputs: dict[str, Any] = {}
                for col in feats:
                    if col in self.predict_vars:
                        kind, var = self.predict_vars[col]
                        raw = var.get()
                        if kind == "num":
                            inputs[col] = float(str(raw).replace(",", ""))
                        else:
                            inputs[col] = raw
                    elif col in self.df.columns:
                        if pd.api.types.is_numeric_dtype(self.df[col]):
                            med = pd.to_numeric(self.df[col], errors="coerce").median()
                            inputs[col] = float(med) if pd.notna(med) else 0.0
                        else:
                            mode = self.df[col].mode()
                            inputs[col] = mode.iloc[0] if len(mode) else ""
                    else:
                        raise ValueError(f"Thiếu feature `{col}` — huấn luyện lại trên dataset hiện tại.")
                result = predict_attrition(model, inputs, feature_columns=feats)
                clear_frame(result_host)
                risk_result_card(
                    result_host, result["probability_pct"], result["risk_band"],
                    result["prediction"], name,
                )
                if self.train_result and "Random Forest" in self.train_result.get("models", {}):
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

    def _ensure_classifier_trained(self, force: bool = False):
        """Trả về (model, feature_columns, name). Train lại nếu thiếu hoặc feature lệch dataset."""
        if not self.target or self.target not in self.df.columns:
            raise ValueError("Chưa có cột target — không train được.")
        if int(self.df[self.target].nunique(dropna=True)) < 2:
            raise ValueError("Target chỉ có 1 lớp — không train được.")

        current_feats, _, _ = self._current_feature_columns()
        current_set = set(current_feats)

        def _usable_memory_model():
            if not (self.train_result and self.eval_result):
                return None
            feats = list(self.train_result.get("feature_columns") or [])
            if not feats:
                return None
            # Cho phép model cũ nếu mọi feature của model vẫn còn trong df
            if any(c not in self.df.columns for c in feats):
                return None
            # Nếu schema sạch hơn (bỏ ID/junk) và lệch nhiều → nên train lại
            if current_set and set(feats) != current_set:
                # chỉ tái dùng khi tập model ⊇ current và phần thừa toàn junk/id
                extra = set(feats) - current_set
                junk = set(self.schema.get("junk_cols") or [])
                id_col = self.schema.get("id_col")
                if id_col:
                    junk.add(id_col)
                if not extra.issubset(junk) or (current_set - set(feats)):
                    return None
            best = self.eval_result["best_model_name"]
            return self.train_result["models"][best], feats, best

        if not force:
            mem = _usable_memory_model()
            if mem is not None:
                return mem

        tr = train_classification_models(self.df, save=True, target=self.target)
        ev = evaluate_all_classifiers(tr["models"], tr["X_test"], tr["y_test"])
        self.train_result, self.eval_result = tr, ev
        self.model_status = ev["best_model_name"]
        meta_path = MODELS_DIR / "feature_meta.joblib"
        meta = {
            "best_model_name": ev["best_model_name"],
            "target": self.target,
            "feature_columns": tr["feature_columns"],
            "numeric_features": tr["numeric_features"],
            "categorical_features": tr["categorical_features"],
        }
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(meta, meta_path)
        best = ev["best_model_name"]
        return tr["models"][best], tr["feature_columns"], best

    def _load_model(self):
        return self._ensure_classifier_trained(force=False)

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
