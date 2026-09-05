"""Tối ưu hiệu năng UI — cache + debounce + matplotlib nhẹ."""

from __future__ import annotations

import hashlib
from typing import Any, Callable

import matplotlib as mpl
import pandas as pd


def tune_matplotlib_fast() -> None:
    """Giảm chi phí render chart (không đổi kết quả số liệu)."""
    mpl.rcParams.update(
        {
            "figure.dpi": 72,
            "savefig.dpi": 72,
            "figure.figsize": (5.5, 3.6),
            "path.simplify": True,
            "path.simplify_threshold": 0.2,
            "agg.path.chunksize": 10000,
            "axes.grid": True,
            "figure.max_open_warning": 30,
        }
    )


def filter_signature(filters: dict[str, Any]) -> str:
    parts = []
    for k in sorted(filters.keys()):
        parts.append(f"{k}={filters[k]}")
    return "|".join(parts)


def df_content_id(df: pd.DataFrame) -> str:
    """ID nhẹ để cache theo dataset (shape + cột + vài checksum)."""
    raw = f"{df.shape}|{tuple(df.columns)}|{df.index[0] if len(df) else ''}|{df.index[-1] if len(df) else ''}"
    return hashlib.md5(raw.encode(), usedforsecurity=False).hexdigest()[:12]


class AnalysisCache:
    """Cache KPI / insight / RQ theo (dataset, filter, target)."""

    def __init__(self) -> None:
        self._store: dict[str, Any] = {}

    def key(self, *parts: Any) -> str:
        return "||".join(str(p) for p in parts)

    def get(self, key: str) -> Any | None:
        return self._store.get(key)

    def set(self, key: str, value: Any) -> Any:
        self._store[key] = value
        return value

    def clear(self) -> None:
        self._store.clear()


def debounce(widget, delay_ms: int, callback: Callable[[], None]) -> Callable[[], None]:
    """Gộp nhiều lần gọi filter thành 1 lần sau delay_ms."""
    state = {"after_id": None}

    def wrapped(*_args, **_kwargs) -> None:
        if state["after_id"] is not None:
            try:
                widget.after_cancel(state["after_id"])
            except Exception:  # noqa: BLE001
                pass
        state["after_id"] = widget.after(delay_ms, callback)

    return wrapped
