"""Module tải dataset nhân sự."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.utils import DEFAULT_DATA_PATH, EXPECTED_COLUMNS, TARGET_COL


def load_dataset(path: str | Path | None = None) -> pd.DataFrame:
    """Đọc CSV nhân sự bằng Pandas.

    Args:
        path: Đường dẫn file CSV. Mặc định dùng dataset gốc của project.

    Returns:
        DataFrame đã đọc.

    Raises:
        FileNotFoundError: Nếu file không tồn tại.
        ValueError: Nếu file rỗng hoặc không đọc được.
    """
    data_path = Path(path) if path is not None else DEFAULT_DATA_PATH
    if not data_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file dữ liệu: {data_path}")

    try:
        df = pd.read_csv(data_path)
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"Không đọc được CSV: {exc}") from exc

    if df.empty:
        raise ValueError("Dataset rỗng.")

    return df


def load_uploaded_dataset(uploaded_file: Any) -> pd.DataFrame:
    """Đọc CSV từ đường dẫn file hoặc file-like object."""
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"File CSV không hợp lệ: {exc}") from exc
    if df.empty:
        raise ValueError("File CSV không có dữ liệu.")
    return df


def get_dataset_info(df: pd.DataFrame) -> dict[str, Any]:
    """Trả về thông tin tổng quan dataset."""
    return {
        "n_rows": int(df.shape[0]),
        "n_cols": int(df.shape[1]),
        "columns": list(df.columns),
        "memory_mb": float(df.memory_usage(deep=True).sum() / (1024**2)),
        "has_target": TARGET_COL in df.columns,
    }


def get_column_types(df: pd.DataFrame) -> pd.DataFrame:
    """Bảng kiểu dữ liệu từng cột."""
    rows = []
    for col in df.columns:
        rows.append(
            {
                "Column": col,
                "Data Type": str(df[col].dtype),
                "Non-Null": int(df[col].notna().sum()),
                "Null": int(df[col].isna().sum()),
                "Unique": int(df[col].nunique(dropna=True)),
            }
        )
    return pd.DataFrame(rows)


def validate_uploaded_schema(df: pd.DataFrame) -> tuple[bool, list[str]]:
    """Kiểm tra schema file upload.

    Returns:
        (is_valid, list_of_errors)
    """
    errors: list[str] = []
    cols = set(df.columns)

    if TARGET_COL not in cols:
        errors.append(
            f"Thiếu cột target bắt buộc `{TARGET_COL}`. "
            "Không thể phân tích nghỉ việc khi thiếu biến mục tiêu."
        )

    missing = [c for c in EXPECTED_COLUMNS if c not in cols]
    if missing:
        errors.append(
            "Thiếu một số cột kỳ vọng: " + ", ".join(missing[:10])
            + ("..." if len(missing) > 10 else "")
        )

    extra = [c for c in df.columns if c not in EXPECTED_COLUMNS]
    warnings_ok = True  # extra columns are allowed but noted
    if extra and not errors:
        # Extra columns không khiến fail cứng, nhưng ghi nhận.
        _ = warnings_ok

    # Target values check
    if TARGET_COL in cols:
        vals = set(df[TARGET_COL].astype(str).str.strip().unique())
        allowed = {"Có", "Không"}
        unexpected = vals - allowed
        if unexpected:
            errors.append(
                f"Giá trị `{TARGET_COL}` không hợp lệ: {sorted(unexpected)}. "
                "Chỉ chấp nhận 'Có' / 'Không'."
            )

    return len(errors) == 0, errors
