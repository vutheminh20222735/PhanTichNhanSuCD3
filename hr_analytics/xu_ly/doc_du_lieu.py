"""Module tải dataset — chỉ từ file người dùng upload."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

COMMON_ENCODINGS = ["utf-8", "utf-8-sig", "latin-1", "cp1258", "cp1252"]


def load_csv(
    path: str | Path,
    encoding: str = "utf-8",
    nrows: int | None = None,
) -> pd.DataFrame:
    """Đọc CSV với encoding chỉ định."""
    data_path = Path(path)
    if not data_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file: {data_path}")
    try:
        df = pd.read_csv(data_path, encoding=encoding, nrows=nrows)
    except UnicodeDecodeError as exc:
        raise ValueError(
            f"Encoding `{encoding}` không đọc được file. Thử utf-8 / latin-1 / cp1258."
        ) from exc
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"Không đọc được CSV: {exc}") from exc
    if df.empty:
        raise ValueError("Dataset rỗng.")
    if df.shape[1] < 2:
        raise ValueError("Dataset cần ít nhất 2 cột.")
    return df


def try_load_csv(path: str | Path, encoding: str = "utf-8") -> tuple[pd.DataFrame, str]:
    """Thử encoding được chọn; nếu fail thì thử các encoding phổ biến."""
    tried = [encoding] + [e for e in COMMON_ENCODINGS if e != encoding]
    last_err: Exception | None = None
    for enc in tried:
        try:
            return load_csv(path, encoding=enc), enc
        except ValueError as exc:
            last_err = exc
            # chỉ fallback khi lỗi encoding
            if "Encoding" not in str(exc) and "codec" not in str(exc).lower():
                raise
    raise ValueError(str(last_err) if last_err else "Không đọc được CSV.")


def load_uploaded_dataset(uploaded_file: Any, encoding: str = "utf-8") -> pd.DataFrame:
    """Tương thích API cũ — đọc path hoặc file-like."""
    if isinstance(uploaded_file, (str, Path)):
        df, _ = try_load_csv(uploaded_file, encoding=encoding)
        return df
    try:
        df = pd.read_csv(uploaded_file, encoding=encoding)
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"File CSV không hợp lệ: {exc}") from exc
    if df.empty:
        raise ValueError("File CSV không có dữ liệu.")
    if df.shape[1] < 2:
        raise ValueError("Dataset cần ít nhất 2 cột.")
    return df


def file_size_bytes(path: str | Path | None) -> int | None:
    if not path:
        return None
    p = Path(path)
    return p.stat().st_size if p.exists() else None
