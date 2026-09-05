"""Huấn luyện mô hình Classification và Regression."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.data.schema_detector import build_schema
from src.preprocessing.transformer import (
    build_preprocessor,
    get_feature_columns,
    map_target,
    prepare_xy,
)
from src.utils import ID_COL, MODELS_DIR, TARGET_COL


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple:
    """Train/test split có stratify."""
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )


def build_logistic_pipeline(
    numeric_features: list[str],
    categorical_features: list[str],
) -> Pipeline:
    """Pipeline Logistic Regression với StandardScaler + OneHotEncoder."""
    preprocessor = build_preprocessor(
        numeric_features, categorical_features, scale_numeric=True
    )
    clf = LogisticRegression(
        max_iter=1000,
        random_state=42,
        class_weight="balanced",
    )
    return Pipeline([("preprocess", preprocessor), ("model", clf)])


def build_rf_pipeline(
    numeric_features: list[str],
    categorical_features: list[str],
    n_estimators: int = 200,
    max_depth: int | None = 12,
    min_samples_split: int = 4,
) -> Pipeline:
    """Pipeline Random Forest (không bắt buộc scale numeric)."""
    preprocessor = build_preprocessor(
        numeric_features, categorical_features, scale_numeric=False
    )
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
    )
    return Pipeline([("preprocess", preprocessor), ("model", clf)])


def train_classification_models(
    df: pd.DataFrame,
    save: bool = True,
    target: str | None = None,
) -> dict[str, Any]:
    """Huấn luyện Logistic Regression và Random Forest.

    Preprocessing chỉ fit trên training data nhờ Pipeline.
    """
    X, y, numeric, categorical = prepare_xy(df, target=target)
    X_train, X_test, y_train, y_test = split_data(X, y)

    lr_pipe = build_logistic_pipeline(numeric, categorical)
    rf_pipe = build_rf_pipeline(numeric, categorical)

    lr_pipe.fit(X_train, y_train)
    rf_pipe.fit(X_train, y_train)

    result: dict[str, Any] = {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "numeric_features": numeric,
        "categorical_features": categorical,
        "models": {
            "Logistic Regression": lr_pipe,
            "Random Forest": rf_pipe,
        },
        "feature_columns": list(X.columns),
    }

    if save:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(lr_pipe, MODELS_DIR / "logistic_regression.joblib")
        joblib.dump(rf_pipe, MODELS_DIR / "random_forest.joblib")
        meta = {
            "numeric_features": numeric,
            "categorical_features": categorical,
            "feature_columns": list(X.columns),
        }
        joblib.dump(meta, MODELS_DIR / "feature_meta.joblib")
        result["saved_paths"] = {
            "logistic": str(MODELS_DIR / "logistic_regression.joblib"),
            "random_forest": str(MODELS_DIR / "random_forest.joblib"),
        }

    return result


def train_salary_regression(
    df: pd.DataFrame,
    save: bool = True,
    target: str | None = None,
) -> dict[str, Any]:
    """Track B: Linear Regression dự đoán thu nhập.

    Không dùng cột thu nhập làm feature. Target phát hiện qua schema nếu không truyền.
    """
    schema = build_schema(df)
    income_col = target or schema["roles"].get("income")
    if not income_col or income_col not in df.columns:
        raise ValueError("Không tìm thấy cột thu nhập để huấn luyện regression.")

    attrition_target = schema.get("target") or TARGET_COL
    id_col = schema.get("id_col") or ID_COL
    exclude = {c for c in [attrition_target, id_col, income_col] if c}
    numeric_all, categorical_all = get_feature_columns(
        df, target=attrition_target if attrition_target in df.columns else None
    )
    numeric = [c for c in numeric_all if c not in exclude]
    categorical = [c for c in categorical_all if c not in exclude]

    feature_cols = numeric + categorical
    if not feature_cols:
        raise ValueError("Không có feature cho salary regression.")

    X = df[feature_cols].copy()
    y = pd.to_numeric(df[income_col], errors="coerce")
    valid = y.notna()
    X, y = X.loc[valid], y.loc[valid]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    preprocessor = build_preprocessor(numeric, categorical, scale_numeric=True)
    pipe = Pipeline(
        [
            ("preprocess", preprocessor),
            ("model", LinearRegression()),
        ]
    )
    pipe.fit(X_train, y_train)

    result: dict[str, Any] = {
        "pipeline": pipe,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "numeric_features": numeric,
        "categorical_features": categorical,
        "target": income_col,
    }

    if save:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        path = MODELS_DIR / "salary_linear_regression.joblib"
        joblib.dump(pipe, path)
        result["saved_path"] = str(path)

    return result


def load_saved_model(name: str) -> Any:
    """Load model đã lưu."""
    mapping = {
        "logistic": MODELS_DIR / "logistic_regression.joblib",
        "random_forest": MODELS_DIR / "random_forest.joblib",
        "salary": MODELS_DIR / "salary_linear_regression.joblib",
    }
    path = mapping.get(name)
    if path is None or not Path(path).exists():
        raise FileNotFoundError(f"Không tìm thấy model `{name}` tại {path}")
    return joblib.load(path)
