"""Module biến đổi dữ liệu — hỗ trợ schema động."""

from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data.schema_detector import build_schema, map_binary_target


def map_target(y: pd.Series) -> pd.Series:
    """Map target binary linh hoạt → 0/1."""
    return map_binary_target(y)


def get_feature_columns(
    df: pd.DataFrame,
    target: str | None = None,
    id_col: str | None = None,
) -> tuple[list[str], list[str]]:
    """Lấy numeric/categorical features theo dtype thực tế."""
    schema = build_schema(df, target=target)
    target = target or schema["target"]
    id_col = id_col or schema["id_col"]
    exclude = {c for c in [target, id_col] if c}
    numeric = [c for c in schema["numeric"] if c not in exclude]
    categorical = [c for c in schema["categorical"] if c not in exclude]
    return numeric, categorical


def build_preprocessor(
    numeric_features: list[str],
    categorical_features: list[str],
    scale_numeric: bool = True,
) -> ColumnTransformer:
    transformers: list[tuple[str, Any, list[str]]] = []
    if numeric_features:
        num_pipe: Any = (
            Pipeline(steps=[("scaler", StandardScaler())])
            if scale_numeric
            else "passthrough"
        )
        transformers.append(("num", num_pipe, numeric_features))
    if categorical_features:
        cat_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        transformers.append(("cat", cat_encoder, categorical_features))
    if not transformers:
        raise ValueError("Không có feature để xây preprocessor.")
    return ColumnTransformer(transformers=transformers, remainder="drop")


def prepare_xy(
    df: pd.DataFrame,
    target: str | None = None,
) -> tuple[pd.DataFrame, pd.Series, list[str], list[str]]:
    schema = build_schema(df, target=target)
    target_col = target or schema["target"]
    if not target_col or target_col not in df.columns:
        raise ValueError("Thiếu cột target.")
    numeric, categorical = get_feature_columns(df, target=target_col, id_col=schema["id_col"])
    if not numeric and not categorical:
        raise ValueError("Không tìm thấy feature để train.")
    feature_cols = numeric + categorical
    X = df[feature_cols].copy()
    y = map_target(df[target_col])
    return X, y, numeric, categorical


def get_feature_names_from_preprocessor(
    preprocessor: ColumnTransformer,
    numeric_features: list[str],
    categorical_features: list[str],
) -> list[str]:
    names: list[str] = []
    if numeric_features:
        names.extend(numeric_features)
    if categorical_features:
        try:
            cat_transformer = preprocessor.named_transformers_["cat"]
            if hasattr(cat_transformer, "get_feature_names_out"):
                names.extend(list(cat_transformer.get_feature_names_out(categorical_features)))
            else:
                names.extend(categorical_features)
        except Exception:  # noqa: BLE001
            names.extend(categorical_features)
    return names
