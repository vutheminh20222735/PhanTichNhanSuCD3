"""Trạng thái dataset trung tâm — mọi page đọc từ đây."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

import pandas as pd


class DatasetStatus(str, Enum):
    EMPTY = "empty"
    LOADING = "loading"
    READY = "ready"
    ERROR = "error"


@dataclass
class DatasetState:
    status: DatasetStatus = DatasetStatus.EMPTY
    error_message: str = ""

    name: str | None = None
    path: str | None = None
    encoding: str = "utf-8"
    file_size_bytes: int | None = None
    loaded_at: datetime | None = None

    df: pd.DataFrame | None = None
    df_cleaned: pd.DataFrame | None = None

    schema: dict[str, Any] = field(default_factory=dict)
    target: str | None = None
    profile: dict[str, Any] = field(default_factory=dict)
    quality_report: dict[str, Any] = field(default_factory=dict)

    train_result: Any = None
    eval_result: Any = None
    salary_eval: Any = None
    model_status: str = "Not trained"
    filter_state: dict[str, str] = field(default_factory=dict)
    cache_store: dict[str, Any] = field(default_factory=dict)

    @property
    def ready(self) -> bool:
        return self.status == DatasetStatus.READY and self.df is not None and not self.df.empty

    @property
    def active_df(self) -> pd.DataFrame:
        if self.df_cleaned is not None and not self.df_cleaned.empty:
            return self.df_cleaned
        if self.df is None:
            return pd.DataFrame()
        return self.df

    @property
    def row_count(self) -> int:
        return 0 if self.df is None else len(self.df)

    @property
    def column_count(self) -> int:
        return 0 if self.df is None else int(self.df.shape[1])

    def clear(self) -> None:
        self.status = DatasetStatus.EMPTY
        self.error_message = ""
        self.name = None
        self.path = None
        self.encoding = "utf-8"
        self.file_size_bytes = None
        self.loaded_at = None
        self.df = None
        self.df_cleaned = None
        self.schema = {}
        self.target = None
        self.profile = {}
        self.quality_report = {}
        self.reset_analysis()

    def reset_analysis(self) -> None:
        self.train_result = None
        self.eval_result = None
        self.salary_eval = None
        self.model_status = "Not trained"
        self.filter_state.clear()
        self.cache_store.clear()

    def activate(
        self,
        df: pd.DataFrame,
        *,
        name: str,
        path: str | None,
        encoding: str,
        file_size_bytes: int | None,
        schema: dict[str, Any],
        target: str | None,
        profile: dict[str, Any],
    ) -> None:
        self.reset_analysis()
        self.status = DatasetStatus.READY
        self.error_message = ""
        self.name = name
        self.path = path
        self.encoding = encoding
        self.file_size_bytes = file_size_bytes
        self.loaded_at = datetime.now()
        self.df = df
        self.df_cleaned = None
        self.schema = schema
        self.target = target
        self.profile = profile
        self.quality_report = {}
