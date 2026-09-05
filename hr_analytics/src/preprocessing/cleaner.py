"""Module làm sạch dữ liệu nhân sự."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.data.validator import check_duplicates, check_missing_values, validate_dataset
from src.utils import OUTLIER_COLUMNS, PROCESSED_DIR


def detect_outliers_iqr(df: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    """Phát hiện outlier bằng phương pháp IQR.

    Không xóa outlier — chỉ báo cáo Q1, Q3, IQR, bounds, count, %.
    """
    cols = columns or OUTLIER_COLUMNS
    rows = []
    for col in cols:
        if col not in df.columns:
            continue
        series = pd.to_numeric(df[col], errors="coerce").dropna()
        if series.empty:
            continue
        q1 = float(series.quantile(0.25))
        q3 = float(series.quantile(0.75))
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        n_out = int(((series < lower) | (series > upper)).sum())
        rows.append(
            {
                "Column": col,
                "Q1": round(q1, 2),
                "Q3": round(q3, 2),
                "IQR": round(iqr, 2),
                "Lower Bound": round(lower, 2),
                "Upper Bound": round(upper, 2),
                "Outlier Count": n_out,
                "Outlier %": round(n_out / len(series) * 100, 2),
            }
        )
    return pd.DataFrame(rows)


def check_data_quality(df: pd.DataFrame) -> dict[str, Any]:
    """Kiểm tra chất lượng tổng hợp trước khi làm sạch."""
    validation = validate_dataset(df)
    outliers = detect_outliers_iqr(df)
    return {
        **validation,
        "outliers": outliers,
    }


def clean_dataset(
    df: pd.DataFrame,
    drop_duplicates: bool = True,
    save: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Làm sạch dataset một cách thận trọng.

    - Không tự ý xóa outlier.
    - Chỉ loại duplicate nếu được bật và thực sự tồn tại.
    - Chuẩn hóa khoảng trắng trên cột chuỗi.
    - Không bịa missing/duplicate.
    """
    report: dict[str, Any] = {
        "rows_before": len(df),
        "missing_before": int(df.isnull().sum().sum()),
        "duplicates_before": int(df.duplicated().sum()),
        "actions": [],
    }

    cleaned = df.copy()

    # Chuẩn hóa chuỗi
    str_cols = list(cleaned.select_dtypes(include=["object", "string", "str"]).columns)
    for col in str_cols:
        cleaned[col] = cleaned[col].astype(str).str.strip()
        report["actions"].append(f"Chuẩn hóa khoảng trắng cột `{col}`.")

    # Missing: nếu không có thì ghi nhận đúng
    missing_df = check_missing_values(cleaned)
    total_missing = int(missing_df["Missing Count"].sum())
    if total_missing == 0:
        report["actions"].append("Không phát hiện missing values.")
    else:
        report["actions"].append(
            f"Phát hiện {total_missing} missing values — giữ nguyên để báo cáo "
            "(không tự ý điền/xóa nếu chưa có quy tắc nghiệp vụ)."
        )

    # Duplicate
    dup = check_duplicates(cleaned)
    if dup["duplicate_count"] == 0:
        report["actions"].append("Không phát hiện duplicate records.")
    elif drop_duplicates:
        cleaned = cleaned.drop_duplicates().reset_index(drop=True)
        report["actions"].append(
            f"Đã loại {dup['duplicate_count']} bản ghi trùng lặp."
        )
    else:
        report["actions"].append(
            f"Phát hiện {dup['duplicate_count']} duplicate nhưng chưa xóa "
            "(drop_duplicates=False)."
        )

    # Outlier: chỉ phát hiện, không xóa
    outliers = detect_outliers_iqr(cleaned)
    report["outliers"] = outliers
    report["actions"].append(
        "Đã phân tích outlier bằng IQR — không tự động xóa outlier."
    )

    report["rows_after"] = len(cleaned)
    report["missing_after"] = int(cleaned.isnull().sum().sum())
    report["duplicates_after"] = int(cleaned.duplicated().sum())

    if save:
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        out_path = PROCESSED_DIR / "dataset_cleaned.csv"
        cleaned.to_csv(out_path, index=False)
        report["saved_path"] = str(out_path)

    return cleaned, report
