"""UI components — presentation layer PeopleRisk AI."""

from __future__ import annotations

import tkinter as tk
import weakref
from tkinter import ttk
from typing import Callable

import customtkinter as ctk
import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from giao_dien.giao_dien_mau import THEME

# Registry dropdown đang sống — đóng khi scroll / đổi trang
_OPTION_MENU_REFS: list[weakref.ref] = []
_DROPDOWN_OPEN_PATCHED = False


def close_all_dropdowns() -> None:
    """Đóng mọi CTkOptionMenu đang mở (scroll / đổi tab)."""
    alive: list[weakref.ref] = []
    for ref in _OPTION_MENU_REFS:
        menu = ref()
        if menu is None:
            continue
        alive.append(ref)
        try:
            dm = getattr(menu, "_dropdown_menu", None)
            if dm is not None:
                dm.close()
            if hasattr(menu, "_close_on_next_click"):
                menu._close_on_next_click = False
        except Exception:  # noqa: BLE001
            pass
    _OPTION_MENU_REFS[:] = alive


def _patch_dropdown_open_once() -> None:
    """Linux: dùng post() thay tk_popup() để đóng/unpost ổn định khi scroll."""
    global _DROPDOWN_OPEN_PATCHED
    if _DROPDOWN_OPEN_PATCHED:
        return
    try:
        from customtkinter.windows.widgets.core_widget_classes.dropdown_menu import DropdownMenu
    except Exception:  # noqa: BLE001
        return

    def _open_fixed(self, x, y) -> None:  # noqa: ANN001
        y = int(y) + int(self._apply_widget_scaling(3))
        x = int(x)
        try:
            self.unpost()
        except Exception:  # noqa: BLE001
            pass
        self.post(x, y)

    DropdownMenu.open = _open_fixed  # type: ignore[method-assign]
    _DROPDOWN_OPEN_PATCHED = True


def _attach_option_menu_behavior(menu: ctk.CTkOptionMenu) -> None:
    """Gắn mở đúng dưới trường + đăng ký để đóng hàng loạt."""
    _patch_dropdown_open_once()
    _OPTION_MENU_REFS.append(weakref.ref(menu))

    def _open_anchored(_event=None) -> None:
        close_all_dropdowns()
        try:
            menu.update_idletasks()
            h = max(int(menu.winfo_height()), int(getattr(menu, "_current_height", 0) or 0), 28)
            x = int(menu.winfo_rootx())
            y = int(menu.winfo_rooty()) + h
            menu._dropdown_menu.open(x, y)
            menu._close_on_next_click = True
        except Exception:  # noqa: BLE001
            # fallback hành vi gốc
            try:
                menu._dropdown_menu.open(
                    menu.winfo_rootx(),
                    menu.winfo_rooty() + menu._apply_widget_scaling(menu._current_height),
                )
                menu._close_on_next_click = True
            except Exception:  # noqa: BLE001
                pass

    menu._open_dropdown_menu = _open_anchored  # type: ignore[method-assign]


def font(size: int = 13, weight: str = "normal") -> ctk.CTkFont:
    try:
        return ctk.CTkFont(family=THEME.font_family, size=size, weight=weight)
    except Exception:  # noqa: BLE001
        return ctk.CTkFont(family=THEME.font_fallback, size=size, weight=weight)


class ScrollableFrame(ctk.CTkScrollableFrame):
    def __init__(self, master, **kwargs):
        kwargs.setdefault("fg_color", THEME.bg)
        kwargs.setdefault("corner_radius", 0)
        super().__init__(master, **kwargs)
        self._wire_close_dropdowns_on_scroll()

    def _wire_close_dropdowns_on_scroll(self) -> None:
        """Khi lướt trang / kéo scrollbar → đóng dropdown đang mở."""

        def _on_scroll(_event=None):
            close_all_dropdowns()

        canvas = getattr(self, "_parent_canvas", None)
        scrollbar = getattr(self, "_scrollbar", None)
        if canvas is not None:
            for seq in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
                canvas.bind(seq, _on_scroll, add="+")
            if scrollbar is not None:
                def _yscroll(first, last):
                    close_all_dropdowns()
                    try:
                        scrollbar.set(first, last)
                    except Exception:  # noqa: BLE001
                        pass

                canvas.configure(yscrollcommand=_yscroll)

        if scrollbar is not None:
            for seq in ("<ButtonPress-1>", "<B1-Motion>", "<ButtonRelease-1>"):
                try:
                    scrollbar.bind(seq, _on_scroll, add="+")
                except Exception:  # noqa: BLE001
                    pass


def clear_frame(frame: tk.Misc) -> None:
    for child in frame.winfo_children():
        child.destroy()


def configure_treeview_style(root: tk.Misc) -> None:
    style = ttk.Style(root)
    try:
        style.theme_use("clam")
    except Exception:  # noqa: BLE001
        pass
    style.configure(
        "HR.Treeview",
        background=THEME.surface,
        foreground=THEME.text,
        fieldbackground=THEME.surface,
        borderwidth=0,
        rowheight=28,
        font=(THEME.font_fallback, 10),
    )
    style.configure(
        "HR.Treeview.Heading",
        background=THEME.surface_alt,
        foreground=THEME.text,
        relief="flat",
        font=(THEME.font_fallback, 10, "bold"),
        borderwidth=0,
    )
    style.map(
        "HR.Treeview",
        background=[("selected", THEME.accent_soft)],
        foreground=[("selected", THEME.brand)],
    )


def status_pill(parent, text: str, kind: str = "info") -> ctk.CTkFrame:
    colors = {
        "success": (THEME.success_soft, THEME.success),
        "danger": (THEME.danger_soft, THEME.danger),
        "warning": (THEME.warning_soft, THEME.warning),
        "info": (THEME.info_soft, THEME.info),
        "neutral": (THEME.surface_alt, THEME.text_secondary),
        "high": (THEME.danger_soft, THEME.danger),
        "medium": (THEME.warning_soft, THEME.warning),
        "low": (THEME.success_soft, THEME.success),
    }
    bg, fg = colors.get(kind.lower(), colors["info"])
    wrap = ctk.CTkFrame(parent, fg_color=bg, corner_radius=16, height=26)
    wrap.pack_propagate(False)
    ctk.CTkLabel(wrap, text=text, text_color=fg, font=font(10, "bold")).pack(padx=10, pady=3)
    return wrap


def make_kpi_card(
    parent,
    title: str,
    value: str,
    subtitle: str = "",
    tone: str = "brand",
) -> ctk.CTkFrame:
    accents = {
        "brand": THEME.brand,
        "accent": THEME.accent,
        "danger": THEME.danger,
        "warning": THEME.warning,
        "success": THEME.success,
        "info": THEME.info,
    }
    accent = accents.get(tone, THEME.brand)
    card = ctk.CTkFrame(
        parent,
        height=96,
        corner_radius=12,
        fg_color=THEME.surface,
        border_width=1,
        border_color=THEME.border,
    )
    card.pack_propagate(False)
    ctk.CTkFrame(card, width=4, corner_radius=0, fg_color=accent).place(x=0, y=0, relheight=1)

    ctk.CTkLabel(
        card, text=title.upper(), font=font(10, "bold"), text_color=THEME.text_muted
    ).pack(anchor="w", padx=(14, 10), pady=(12, 0))
    ctk.CTkLabel(card, text=value, font=font(20, "bold"), text_color=THEME.text).pack(
        anchor="w", padx=(14, 10), pady=(2, 0)
    )
    if subtitle:
        ctk.CTkLabel(
            card, text=subtitle, font=font(10), text_color=THEME.text_secondary
        ).pack(anchor="w", padx=(14, 10), pady=(2, 8))
    return card


def panel(parent, title: str = "", subtitle: str = "") -> ctk.CTkFrame:
    box = ctk.CTkFrame(
        parent,
        fg_color=THEME.surface,
        corner_radius=12,
        border_width=1,
        border_color=THEME.border,
    )
    if title:
        head = ctk.CTkFrame(box, fg_color="transparent")
        head.pack(fill="x", padx=14, pady=(12, 2))
        ctk.CTkLabel(head, text=title, font=font(14, "bold"), text_color=THEME.text).pack(
            anchor="w"
        )
        if subtitle:
            ctk.CTkLabel(
                head, text=subtitle, font=font(11), text_color=THEME.text_secondary
            ).pack(anchor="w")
    return box


def section_title(parent, text: str, subtitle: str = "") -> ctk.CTkFrame:
    wrap = ctk.CTkFrame(parent, fg_color="transparent")
    wrap.pack(fill="x", padx=6, pady=(12, 4))
    ctk.CTkLabel(wrap, text=text, font=font(15, "bold"), text_color=THEME.text).pack(anchor="w")
    if subtitle:
        ctk.CTkLabel(
            wrap, text=subtitle, font=font(11), text_color=THEME.text_secondary
        ).pack(anchor="w")
    return wrap


def body_text(parent, text: str, wrap: int = 1100, muted: bool = False) -> ctk.CTkLabel:
    lbl = ctk.CTkLabel(
        parent,
        text=text,
        justify="left",
        wraplength=wrap,
        font=font(12),
        text_color=THEME.text_secondary if muted else THEME.text,
    )
    lbl.pack(anchor="w", padx=12, pady=3)
    return lbl


def metric_chip(parent, label: str, value: str) -> ctk.CTkFrame:
    chip = ctk.CTkFrame(
        parent, fg_color=THEME.surface_alt, corner_radius=8, border_width=0
    )
    ctk.CTkLabel(chip, text=label, font=font(9, "bold"), text_color=THEME.text_muted).pack(
        anchor="w", padx=10, pady=(6, 0)
    )
    ctk.CTkLabel(chip, text=value, font=font(13, "bold"), text_color=THEME.text).pack(
        anchor="w", padx=10, pady=(0, 6)
    )
    return chip


def insight_card(
    parent,
    index: int,
    group: str,
    title: str,
    insight: str,
    recommendation: str | None = None,
    evidence: str | None = None,
    difference: str | None = None,
    severity: str | None = None,
) -> ctk.CTkFrame:
    card = ctk.CTkFrame(
        parent,
        fg_color=THEME.surface,
        corner_radius=12,
        border_width=1,
        border_color=THEME.border,
    )
    card.pack(fill="x", padx=6, pady=6)

    top = ctk.CTkFrame(card, fg_color="transparent")
    top.pack(fill="x", padx=14, pady=(12, 4))
    ctk.CTkLabel(
        top,
        text=f"#{index:02d}  {group.upper()}",
        font=font(10, "bold"),
        text_color=THEME.accent,
        fg_color=THEME.accent_soft,
        corner_radius=8,
        padx=8,
        pady=3,
    ).pack(side="left")
    if severity:
        status_pill(top, severity.upper(), severity.lower()).pack(side="right")

    ctk.CTkLabel(card, text=title, font=font(14, "bold"), text_color=THEME.text).pack(
        anchor="w", padx=14, pady=(2, 4)
    )

    if evidence or difference:
        meta = ctk.CTkFrame(card, fg_color="transparent")
        meta.pack(fill="x", padx=10, pady=(0, 4))
        if evidence:
            metric_chip(meta, "EVIDENCE", evidence).pack(side="left", padx=4)
        if difference:
            metric_chip(meta, "DIFFERENCE", difference).pack(side="left", padx=4)

    ctk.CTkLabel(
        card,
        text=insight,
        font=font(12),
        text_color=THEME.text_secondary,
        wraplength=1000,
        justify="left",
    ).pack(anchor="w", padx=14, pady=(2, 8))

    if recommendation:
        rec = ctk.CTkFrame(card, fg_color=THEME.info_soft, corner_radius=8)
        rec.pack(fill="x", padx=14, pady=(0, 12))
        ctk.CTkLabel(
            rec, text="Recommendation", font=font(10, "bold"), text_color=THEME.info
        ).pack(anchor="w", padx=10, pady=(8, 0))
        ctk.CTkLabel(
            rec,
            text=recommendation,
            font=font(12),
            text_color=THEME.text,
            wraplength=960,
            justify="left",
        ).pack(anchor="w", padx=10, pady=(2, 8))
    return card


def risk_result_card(
    parent,
    pct: float,
    band: str,
    prediction: str,
    model_name: str,
) -> ctk.CTkFrame:
    if band == "High":
        tone_bg, tone_fg = THEME.danger_soft, THEME.danger
    elif band == "Medium":
        tone_bg, tone_fg = THEME.warning_soft, THEME.warning
    else:
        tone_bg, tone_fg = THEME.success_soft, THEME.success

    card = ctk.CTkFrame(
        parent,
        fg_color=THEME.surface,
        corner_radius=14,
        border_width=1,
        border_color=THEME.border,
    )
    card.pack(fill="x", padx=6, pady=8)

    score = ctk.CTkFrame(card, fg_color=tone_bg, corner_radius=12, width=200, height=170)
    score.pack(side="left", padx=16, pady=16)
    score.pack_propagate(False)
    ctk.CTkLabel(score, text="RISK SCORE", font=font(11, "bold"), text_color=tone_fg).pack(
        pady=(28, 0)
    )
    ctk.CTkLabel(score, text=f"{pct:.1f}%", font=font(34, "bold"), text_color=tone_fg).pack()
    ctk.CTkLabel(score, text=band.upper(), font=font(14, "bold"), text_color=tone_fg).pack()

    right = ctk.CTkFrame(card, fg_color="transparent")
    right.pack(side="left", fill="both", expand=True, padx=(4, 18), pady=18)
    ctk.CTkLabel(right, text="Risk Level", font=font(11), text_color=THEME.text_muted).pack(
        anchor="w"
    )
    ctk.CTkLabel(right, text=band.upper(), font=font(22, "bold"), text_color=tone_fg).pack(
        anchor="w"
    )
    ctk.CTkLabel(right, text="Prediction", font=font(11), text_color=THEME.text_muted).pack(
        anchor="w", pady=(10, 0)
    )
    pred_text = (
        "Có khả năng nghỉ việc" if prediction == "Có" else "Khả năng ở lại cao hơn"
    )
    ctk.CTkLabel(right, text=pred_text, font=font(16, "bold"), text_color=THEME.text).pack(
        anchor="w"
    )
    ctk.CTkLabel(
        right, text=f"Model: {model_name}", font=font(12), text_color=THEME.text_secondary
    ).pack(anchor="w", pady=(10, 0))

    bar = ctk.CTkProgressBar(right, width=380, height=12, progress_color=tone_fg)
    bar.pack(anchor="w", pady=(12, 4))
    bar.set(max(0.0, min(1.0, pct / 100)))
    ctk.CTkLabel(
        right,
        text="Prototype thresholds — không phải tiêu chuẩn HR thực tế.  (0–30 Low · 30–60 Medium · 60–100 High)",
        font=font(10),
        text_color=THEME.text_muted,
        wraplength=520,
        justify="left",
    ).pack(anchor="w")
    return card


def show_dataframe(
    parent,
    df: pd.DataFrame,
    height: int = 200,
    page_size: int = 20,
    enable_search: bool = False,
) -> ctk.CTkFrame:
    """Bảng có pagination, chiều cao kiểm soát."""
    wrap = ctk.CTkFrame(
        parent,
        fg_color=THEME.surface,
        corner_radius=10,
        border_width=1,
        border_color=THEME.border,
    )
    wrap.pack(fill="both", expand=True, padx=6, pady=4)

    state = {"page": 0, "query": "", "df": df}

    toolbar = ctk.CTkFrame(wrap, fg_color="transparent")
    toolbar.pack(fill="x", padx=8, pady=(8, 2))
    info = ctk.CTkLabel(toolbar, text="", font=font(10), text_color=THEME.text_muted)
    info.pack(side="left")

    search_var = ctk.StringVar(value="")
    if enable_search:
        entry = ctk.CTkEntry(
            toolbar,
            textvariable=search_var,
            placeholder_text="Tìm trong bảng...",
            width=200,
            height=28,
        )
        entry.pack(side="right", padx=4)

    tree_host = tk.Frame(wrap, bg=THEME.surface)
    tree_host.pack(fill="both", expand=True, padx=8, pady=4)

    cols = list(df.columns)
    tree = ttk.Treeview(
        tree_host,
        columns=cols,
        show="headings",
        height=max(5, min(12, height // 28)),
        style="HR.Treeview",
    )
    vsb = ttk.Scrollbar(tree_host, orient="vertical", command=tree.yview)
    hsb = ttk.Scrollbar(tree_host, orient="horizontal", command=tree.xview)
    tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
    for col in cols:
        tree.heading(col, text=str(col))
        tree.column(col, width=max(90, min(180, 10 * len(str(col)) + 36)), anchor="center")
    tree.grid(row=0, column=0, sticky="nsew")
    vsb.grid(row=0, column=1, sticky="ns")
    hsb.grid(row=1, column=0, sticky="ew")
    tree_host.grid_rowconfigure(0, weight=1)
    tree_host.grid_columnconfigure(0, weight=1)

    nav = ctk.CTkFrame(wrap, fg_color="transparent")
    nav.pack(fill="x", padx=8, pady=(0, 8))
    prev_btn = ctk.CTkButton(nav, text="‹ Trước", width=80, height=28)
    next_btn = ctk.CTkButton(nav, text="Sau ›", width=80, height=28)
    prev_btn.pack(side="left")
    next_btn.pack(side="left", padx=6)

    def filtered() -> pd.DataFrame:
        q = state["query"].strip().lower()
        base = state["df"]
        if not q:
            return base
        mask = False
        for c in base.columns:
            mask = mask | base[c].astype(str).str.lower().str.contains(q, na=False)
        return base[mask]

    def render() -> None:
        view = filtered()
        total = len(view)
        pages = max(1, (total + page_size - 1) // page_size)
        state["page"] = max(0, min(state["page"], pages - 1))
        start = state["page"] * page_size
        end = min(start + page_size, total)
        tree.delete(*tree.get_children())
        chunk = view.iloc[start:end]
        records = chunk.to_numpy().tolist()
        col_idx = list(range(len(cols)))
        for i, row_vals in enumerate(records):
            values = []
            for j in col_idx:
                val = row_vals[j]
                if val is None or (isinstance(val, float) and pd.isna(val)):
                    values.append("")
                elif isinstance(val, float):
                    values.append(f"{val:.4g}" if abs(val) < 1e6 else f"{val:,.0f}")
                else:
                    text = str(val)
                    values.append(text if len(text) < 40 else text[:37] + "...")
            tag = "odd" if i % 2 else "even"
            tree.insert("", "end", values=values, tags=(tag,))
        tree.tag_configure("even", background=THEME.surface)
        tree.tag_configure("odd", background=THEME.surface_alt)
        info.configure(text=f"{start + 1 if total else 0}–{end} / {total} dòng  ·  trang {state['page'] + 1}/{pages}")

    def go_prev() -> None:
        state["page"] -= 1
        render()

    def go_next() -> None:
        state["page"] += 1
        render()

    def on_search(*_args) -> None:
        state["query"] = search_var.get()
        state["page"] = 0
        render()

    prev_btn.configure(command=go_prev, fg_color=THEME.surface_alt, text_color=THEME.text, hover_color=THEME.border)
    next_btn.configure(command=go_next, fg_color=THEME.surface_alt, text_color=THEME.text, hover_color=THEME.border)
    if enable_search:
        search_var.trace_add("write", on_search)
    render()
    return wrap


def embed_figure(
    parent,
    fig: Figure | None,
    height: int = 300,
    title: str = "",
    note: str = "",
    error: str | None = None,
) -> FigureCanvasTkAgg | None:
    box = panel(parent, title=title)
    box.pack(fill="both", expand=True, padx=4, pady=4)
    if error:
        ctk.CTkLabel(
            box,
            text=f"Không render được biểu đồ: {error}",
            font=font(12),
            text_color=THEME.danger,
            wraplength=480,
            justify="left",
        ).pack(padx=14, pady=20, anchor="w")
        return None
    if fig is None:
        ctk.CTkLabel(
            box, text="Đang tải biểu đồ...", font=font(12), text_color=THEME.text_muted
        ).pack(padx=14, pady=20)
        return None

    holder = ctk.CTkFrame(box, fg_color=THEME.surface, height=height)
    holder.pack(fill="both", expand=True, padx=8, pady=(2, 6))
    fig.patch.set_facecolor(THEME.surface)
    for ax in fig.get_axes():
        ax.set_facecolor(THEME.surface)
        ax.tick_params(colors=THEME.text_secondary, labelsize=8)
        ax.title.set_color(THEME.text)
        ax.xaxis.label.set_color(THEME.text_secondary)
        ax.yaxis.label.set_color(THEME.text_secondary)
        for spine in ax.spines.values():
            spine.set_color(THEME.border)
    try:
        canvas = FigureCanvasTkAgg(fig, master=holder)
        canvas.draw_idle()
        canvas.get_tk_widget().pack(fill="both", expand=True)
    except Exception as exc:  # noqa: BLE001
        ctk.CTkLabel(
            box, text=f"Lỗi render: {exc}", font=font(12), text_color=THEME.danger
        ).pack(padx=14, pady=12)
        return None
    if note:
        ctk.CTkLabel(
            box,
            text=note,
            font=font(11),
            text_color=THEME.text_secondary,
            wraplength=520,
            justify="left",
        ).pack(anchor="w", padx=14, pady=(0, 10))
    return canvas


def safe_chart(builder) -> tuple[Figure | None, str, str | None]:
    """Gọi builder chart; trả (fig, comment, error)."""
    try:
        result = builder()
        if isinstance(result, tuple) and len(result) >= 2:
            return result[0], str(result[1]), None
        return result, "", None
    except Exception as exc:  # noqa: BLE001
        return None, "", str(exc)


def primary_button(parent, text: str, command: Callable, **kwargs) -> ctk.CTkButton:
    kwargs.setdefault("fg_color", THEME.accent)
    kwargs.setdefault("hover_color", THEME.accent_hover)
    kwargs.setdefault("text_color", "#FFFFFF")
    kwargs.setdefault("corner_radius", 8)
    kwargs.setdefault("height", 38)
    kwargs.setdefault("font", font(12, "bold"))
    return ctk.CTkButton(parent, text=text, command=command, **kwargs)


<<<<<<< HEAD
FIELD_BORDER_FOCUS = "#2B6174"
FIELD_HEIGHT = 28
FIELD_RADIUS = 9


class _FieldOptionMenu(ctk.CTkFrame):
    """Option menu with the same border treatment as the text entry."""

    def __init__(self, parent, values: list[str], variable=None, **kwargs) -> None:
        width = kwargs.pop("width", 140)
        height = kwargs.pop("height", FIELD_HEIGHT)
        border_color = kwargs.pop("border_color", THEME.border)
        kwargs.pop("border_width", None)

        super().__init__(
            parent,
            width=width,
            height=height,
            fg_color=border_color,
            corner_radius=FIELD_RADIUS,
        )
        self._border_color = border_color
        self._values = list(values or [""])
        self._popup = None
        self._popup_scroll_bindings: list[str] = []
        self.grid_propagate(False)
        kwargs.setdefault("fg_color", THEME.surface)
        kwargs.setdefault("button_color", THEME.surface)
        kwargs.setdefault("button_hover_color", THEME.surface_alt)
        kwargs.setdefault("text_color", THEME.text)
        kwargs.setdefault("dropdown_fg_color", THEME.surface)
        kwargs.setdefault("dropdown_hover_color", THEME.surface_alt)
        kwargs.setdefault("dropdown_text_color", THEME.text)
        kwargs.setdefault("font", font(12))
        self._inner = ctk.CTkOptionMenu(
            self,
            values=self._values,
            variable=variable,
            width=max(1, width - 2),
            height=max(1, height - 2),
            corner_radius=max(1, FIELD_RADIUS - 1),
            **kwargs,
        )
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)
        self._inner.grid(row=0, column=0, padx=1, pady=1, sticky="nsew")
        self._inner._canvas.bind("<Button-1>", self._open_dropdown)
        self._inner._text_label.bind("<Button-1>", self._open_dropdown)

    def _focus_field(self, _event=None) -> None:
        self.configure(fg_color=FIELD_BORDER_FOCUS)

    def _open_dropdown(self, _event=None) -> None:
        self._focus_field()
        if self._popup is not None and self._popup.winfo_exists():
            self._close_popup()
            return

        self.update_idletasks()
        width = self.winfo_width()
        x = self.winfo_rootx()
        y = self.winfo_rooty() + self.winfo_height()
        row_height = 40

        popup = ctk.CTkToplevel(self)
        self._popup = popup
        popup.withdraw()
        popup.overrideredirect(True)
        popup.transient(self.winfo_toplevel())
        window_scaling = popup._get_window_scaling()
        width = max(1, round(width / window_scaling))
        max_height = max(120, round((self.winfo_screenheight() - y - 12) / window_scaling))
        popup_height = min(len(self._values) * row_height + 2, max_height)
        popup.geometry(f"{width}x{popup_height}+{x}+{y}")
        popup.configure(
            fg_color=THEME.surface,
            border_width=1,
            border_color=THEME.border,
            corner_radius=FIELD_RADIUS,
        )

        host = ctk.CTkScrollableFrame(
            popup,
            fg_color=THEME.surface,
            corner_radius=FIELD_RADIUS - 1,
            scrollbar_fg_color=THEME.surface,
            scrollbar_button_color=THEME.border,
            scrollbar_button_hover_color=THEME.text_muted,
        )
        host.pack(fill="both", expand=True, padx=1, pady=1)
        current = self.get()

        for value in self._values:
            selected = value == current
            ctk.CTkButton(
                host,
                text=value,
                anchor="w",
                height=row_height,
                corner_radius=0,
                border_width=0,
                border_spacing=16,
                fg_color=THEME.accent_soft if selected else THEME.surface,
                hover_color=THEME.surface_alt,
                text_color=THEME.brand if selected else THEME.text,
                font=font(12, "bold" if selected else "normal"),
                command=lambda choice=value: self._select_value(choice),
            ).pack(fill="x", expand=True)

        popup.bind("<Escape>", self._close_popup)
        for sequence in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
            self._popup_scroll_bindings.append(
                popup.bind_class("all", sequence, self._close_popup, add="+")
            )
        popup.deiconify()
        popup.lift()
        popup.focus_force()

    def _close_popup(self, _event=None) -> None:
        if self._popup is None:
            return
        for sequence, binding_id in zip(
            ("<MouseWheel>", "<Button-4>", "<Button-5>"),
            self._popup_scroll_bindings,
        ):
            self._popup.unbind_class("all", sequence)
        self._popup_scroll_bindings.clear()
        if self._popup.winfo_exists():
            self._popup.destroy()
        self._popup = None

    def _select_value(self, value: str) -> None:
        self._inner.set(value)
        command = self._inner.cget("command")
        if command is not None:
            command(value)
        self._close_popup()

    def configure(self, **kwargs):
        border_color = kwargs.pop("border_color", None)
        if border_color is not None:
            self._border_color = border_color
            kwargs.setdefault("fg_color", border_color)
        if "command" in kwargs:
            self._inner.configure(command=kwargs.pop("command"))
        if "values" in kwargs:
            self._values = list(kwargs.pop("values") or [""])
            self._inner.configure(values=self._values)
        return super().configure(**kwargs)

    def cget(self, attribute_name):
        if attribute_name == "command":
            return self._inner.cget("command")
        if attribute_name == "border_color":
            return self._border_color
        return super().cget(attribute_name)

    def set(self, value: str) -> None:
        self._inner.set(value)

    def get(self) -> str:
        return self._inner.get()


def option_menu(parent, values: list[str], variable=None, **kwargs) -> _FieldOptionMenu:
    """Dropdown đồng bộ viền với ô nhập liệu."""
    kwargs.setdefault("height", FIELD_HEIGHT)
    kwargs.setdefault("border_color", THEME.border)
    return _FieldOptionMenu(parent, values=values, variable=variable, **kwargs)
=======
def option_menu(parent, values: list[str], variable=None, **kwargs) -> ctk.CTkFrame:
    """Dropdown nền trắng, mũi tên đen, có viền rõ."""
    height = int(kwargs.pop("height", 30))
    width = kwargs.pop("width", None)
    corner = int(kwargs.pop("corner_radius", 8))

    kwargs.setdefault("fg_color", "#FFFFFF")
    kwargs.setdefault("button_color", "#F3F6F8")
    kwargs.setdefault("button_hover_color", "#E8EEF2")
    kwargs.setdefault("text_color", "#111111")
    kwargs.setdefault("text_color_disabled", "#999999")
    kwargs.setdefault("dropdown_fg_color", THEME.surface)
    kwargs.setdefault("dropdown_hover_color", THEME.surface_alt)
    kwargs.setdefault("dropdown_text_color", THEME.text)
    kwargs.setdefault("font", font(12))
    kwargs.setdefault("dropdown_font", font(12))

    wrap_kw: dict = {
        "fg_color": "#FFFFFF",
        "border_width": 1,
        "border_color": THEME.border,
        "corner_radius": corner,
        "height": height,
    }
    if width is not None:
        wrap_kw["width"] = width
    wrap = ctk.CTkFrame(parent, **wrap_kw)
    # Chỉ khóa kích thước khi có width cố định; còn lại để pack(fill="x") giãn được
    if width is not None:
        wrap.pack_propagate(False)

    menu_kw = dict(kwargs)
    menu_kw["height"] = max(height - 2, 24)
    menu_kw["corner_radius"] = max(corner - 2, 4)
    if width is not None:
        menu_kw["width"] = max(int(width) - 2, 40)
    menu = ctk.CTkOptionMenu(wrap, values=values or [""], variable=variable, **menu_kw)
    menu.pack(fill="both", expand=True, padx=1, pady=1)
    _attach_option_menu_behavior(menu)

    wrap.menu = menu  # type: ignore[attr-defined]
    return wrap
>>>>>>> 52081888a1b82ad484f491144745e6fc162ba71d


def text_entry(parent, textvariable=None, **kwargs) -> ctk.CTkEntry:
    """Ô nhập liệu dùng chung cho các form."""
    kwargs.setdefault("fg_color", THEME.surface)
    kwargs.setdefault("border_width", 1)
    kwargs.setdefault("border_color", THEME.border)
    kwargs.setdefault("text_color", THEME.text)
    kwargs.setdefault("placeholder_text_color", THEME.text_muted)
    kwargs.setdefault("font", font(12))
    kwargs.setdefault("height", FIELD_HEIGHT)
    kwargs.setdefault("corner_radius", FIELD_RADIUS)
    entry = ctk.CTkEntry(parent, textvariable=textvariable, **kwargs)
    entry.bind(
        "<FocusIn>",
        lambda _event: entry.configure(border_color=FIELD_BORDER_FOCUS),
        add="+",
    )
    entry.bind(
        "<FocusOut>",
        lambda _event: entry.configure(border_color=THEME.border),
        add="+",
    )
    return entry


def secondary_button(parent, text: str, command: Callable, **kwargs) -> ctk.CTkButton:
    kwargs.setdefault("fg_color", THEME.surface_alt)
    kwargs.setdefault("hover_color", THEME.border)
    kwargs.setdefault("text_color", THEME.text)
    kwargs.setdefault("border_width", 1)
    kwargs.setdefault("border_color", THEME.border)
    kwargs.setdefault("corner_radius", 8)
    kwargs.setdefault("height", 36)
    kwargs.setdefault("font", font(12))
    return ctk.CTkButton(parent, text=text, command=command, **kwargs)


def quality_status_card(parent, title: str, count: int, ok_label: str) -> ctk.CTkFrame:
    ok = count == 0
    bg = THEME.success_soft if ok else THEME.warning_soft
    fg = THEME.success if ok else THEME.warning
    card = ctk.CTkFrame(
        parent, fg_color=bg, corner_radius=10, width=180, height=88, border_width=0
    )
    card.pack_propagate(False)
    icon = "✓" if ok else "⚠"
    text = f"{icon} {count} {title}" if not ok else f"{icon} {ok_label}"
    ctk.CTkLabel(card, text=text, font=font(14, "bold"), text_color=fg).pack(
        expand=True, padx=10
    )
    return card


def rq_block(parent, question: str) -> ctk.CTkFrame:
    """Khung Research Question: Question → Chart → Metrics → Insight."""
    box = panel(parent, question)
    box.pack(fill="x", padx=6, pady=8)
    return box
