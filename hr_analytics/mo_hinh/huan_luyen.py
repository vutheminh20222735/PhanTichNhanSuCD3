"""Huấn luyện mô hình Classification và Regression (multi-model + CV)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline

from xu_ly.tim_cot_du_lieu import build_schema
from xu_ly.chuan_bi_du_lieu import (
    build_preprocessor,
    get_feature_columns,
    prepare_xy,
)
from xu_ly.tien_ich import MODELS_DIR

try:
    from xgboost import XGBClassifier

    HAS_XGBOOST = True
except Exception:  # noqa: BLE001
    XGBClassifier = None  # type: ignore[misc, assignment]
    HAS_XGBOOST = False


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


def build_gb_pipeline(
    numeric_features: list[str],
    categorical_features: list[str],
) -> Pipeline:
    """Pipeline Gradient Boosting."""
    preprocessor = build_preprocessor(
        numeric_features, categorical_features, scale_numeric=False
    )
    clf = GradientBoostingClassifier(
        n_estimators=150,
        learning_rate=0.08,
        max_depth=3,
        random_state=42,
    )
    return Pipeline([("preprocess", preprocessor), ("model", clf)])


def build_xgb_pipeline(
    numeric_features: list[str],
    categorical_features: list[str],
    y_train: pd.Series | None = None,
) -> Pipeline:
    """Pipeline XGBoost (nếu đã cài xgboost)."""
    if not HAS_XGBOOST or XGBClassifier is None:
        raise ImportError("xgboost chưa được cài đặt.")
    preprocessor = build_preprocessor(
        numeric_features, categorical_features, scale_numeric=False
    )
    # Cân bằng lớp: scale_pos_weight ≈ neg/pos
    scale_pos_weight = 1.0
    if y_train is not None:
        pos = float((y_train == 1).sum())
        neg = float((y_train == 0).sum())
        if pos > 0:
            scale_pos_weight = max(neg / pos, 1.0)
    clf = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.08,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        eval_metric="logloss",
        scale_pos_weight=scale_pos_weight,
    )
    return Pipeline([("preprocess", preprocessor), ("model", clf)])


def build_classifier_zoo(
    numeric: list[str],
    categorical: list[str],
    y_train: pd.Series | None = None,
) -> dict[str, Pipeline]:
    """Tạo bộ mô hình so sánh: LR, RF, GB, (XGB nếu có)."""
    models: dict[str, Pipeline] = {
        "Logistic Regression": build_logistic_pipeline(numeric, categorical),
        "Random Forest": build_rf_pipeline(numeric, categorical),
        "Gradient Boosting": build_gb_pipeline(numeric, categorical),
    }
    if HAS_XGBOOST:
        try:
            models["XGBoost"] = build_xgb_pipeline(numeric, categorical, y_train=y_train)
        except Exception:  # noqa: BLE001
            pass
    return models


def run_cross_validation(
    models: dict[str, Any],
    X: pd.DataFrame,
    y: pd.Series,
    *,
    n_splits: int = 5,
    random_state: int = 42,
) -> pd.DataFrame:
    """Stratified K-Fold CV — trả bảng mean±std theo model."""
    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc",
    }
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    rows: list[dict[str, Any]] = []
    for name, model in models.items():
        scores = cross_validate(
            model,
            X,
            y,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            error_score="raise",
        )
        row: dict[str, Any] = {"Model": name}
        for key in scoring:
            vals = scores[f"test_{key}"]
            row[f"CV_{key.capitalize()}"] = round(float(np.mean(vals)), 4)
            row[f"CV_{key.capitalize()}_std"] = round(float(np.std(vals)), 4)
        rows.append(row)
    return pd.DataFrame(rows)


def train_classification_models(
    df: pd.DataFrame,
    save: bool = True,
    target: str | None = None,
    *,
    run_cv: bool = True,
    cv_folds: int = 5,
) -> dict[str, Any]:
    """Huấn luyện LR / RF / GB / (XGB) + tùy chọn Cross-Validation.

    Preprocessing chỉ fit trên training data nhờ Pipeline.
    """
    X, y, numeric, categorical = prepare_xy(df, target=target)
    X_train, X_test, y_train, y_test = split_data(X, y)

    models = build_classifier_zoo(numeric, categorical, y_train=y_train)
    for pipe in models.values():
        pipe.fit(X_train, y_train)

    cv_table = None
    if run_cv:
        # CV trên full X/y với pipeline mới (không tái dùng model đã fit)
        cv_models = build_classifier_zoo(numeric, categorical, y_train=y)
        try:
            cv_table = run_cross_validation(cv_models, X, y, n_splits=cv_folds)
        except Exception:  # noqa: BLE001
            cv_table = None

    result: dict[str, Any] = {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "numeric_features": numeric,
        "categorical_features": categorical,
        "models": models,
        "feature_columns": list(X.columns),
        "cv_table": cv_table,
        "cv_folds": cv_folds if cv_table is not None else None,
        "has_xgboost": HAS_XGBOOST and "XGBoost" in models,
    }

    if save:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        saved: dict[str, str] = {}
        name_map = {
            "Logistic Regression": "logistic_regression.joblib",
            "Random Forest": "random_forest.joblib",
            "Gradient Boosting": "gradient_boosting.joblib",
            "XGBoost": "xgboost.joblib",
        }
        for name, pipe in models.items():
            fname = name_map.get(name)
            if not fname:
                continue
            path = MODELS_DIR / fname
            joblib.dump(pipe, path)
            saved[name] = str(path)
        meta = {
            "numeric_features": numeric,
            "categorical_features": categorical,
            "feature_columns": list(X.columns),
            "model_names": list(models.keys()),
        }
        joblib.dump(meta, MODELS_DIR / "feature_meta.joblib")
        result["saved_paths"] = saved

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

    attrition_target = schema.get("target")
    id_col = schema.get("id_col")
    exclude = {c for c in [attrition_target, id_col, income_col] if c}
    numeric_all, categorical_all = get_feature_columns(
        df, target=attrition_target if attrition_target and attrition_target in df.columns else None
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
        "gradient_boosting": MODELS_DIR / "gradient_boosting.joblib",
        "xgboost": MODELS_DIR / "xgboost.joblib",
        "salary": MODELS_DIR / "salary_linear_regression.joblib",
    }
    path = mapping.get(name)
    if path is None or not Path(path).exists():
        raise FileNotFoundError(f"Không tìm thấy model `{name}` tại {path}")
    return joblib.load(path)
