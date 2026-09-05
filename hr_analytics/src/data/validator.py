"""Module kiểm tra chất lượng dữ liệu nhân sự."""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.utils import CATEGORICAL_COLUMNS, NUMERIC_COLUMNS, TARGET_COL


def check_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Kiểm tra missing values theo cột."""
    missing = df.isnull().sum()
    pct = (missing / len(df) * 100) if len(df) else missing
    result = pd.DataFrame(
        {
            "Column": missing.index,
            "Missing Count": missing.values,
            "Missing %": [round(float(x), 2) for x in pct.values],
        }
    )
    return result.sort_values("Missing Count", ascending=False).reset_index(drop=True)


def check_duplicates(df: pd.DataFrame) -> dict[str, Any]:
    """Kiểm tra duplicate records."""
    n_dup = int(df.duplicated().sum())
    return {
        "duplicate_count": n_dup,
        "duplicate_pct": round(n_dup / len(df) * 100, 2) if len(df) else 0.0,
        "message": (
            "Không phát hiện duplicate records."
            if n_dup == 0
            else f"Phát hiện {n_dup} bản ghi trùng lặp."
        ),
    }


def check_data_types(df: pd.DataFrame) -> pd.DataFrame:
    """Kiểm tra datatype từng cột."""
    return pd.DataFrame(
        {
            "Column": df.columns,
            "Data Type": [str(df[c].dtype) for c in df.columns],
        }
    )


def get_categorical_uniques(df: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """Số lượng và danh sách category cho biến categorical."""
    result: dict[str, dict[str, Any]] = {}
    cols = [c for c in CATEGORICAL_COLUMNS + [TARGET_COL] if c in df.columns]
    for col in cols:
        uniques = sorted(df[col].dropna().astype(str).unique().tolist())
        result[col] = {
            "n_categories": len(uniques),
            "categories": uniques,
        }
    return result


def check_invalid_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Kiểm tra giá trị không hợp lệ trên biến số.

    Quy tắc cơ bản (theo ngữ cảnh HR):
    - Tuổi: 16–70
    - Thu nhập, khoảng cách, năm kinh nghiệm: >= 0
    - Thang đo hài lòng / hiệu suất: 1–5
    - Phần trăm tăng lương: >= 0
    """
    rules: dict[str, tuple[float | None, float | None]] = {
        "Tuoi": (16, 70),
        "ThuNhapHangThang_VND": (0, None),
        "PhanTramTangLuong": (0, None),
        "KhoangCachNha_Km": (0, None),
        "HaiLong_CongViec": (1, 5),
        "HaiLong_MoiTruong": (1, 5),
        "CanBangCongViec_CuocSong": (1, 5),
        "MucDoGanKetCongViec": (1, 5),
        "HaiLong_QuanHe": (1, 5),
        "DanhGiaHieuSuat": (1, 5),
        "SoLanDaoTao_Nam": (0, None),
        "SoCongTyDaLam": (0, None),
        "TongNamKinhNghiem": (0, None),
        "SoNamTaiCongTy": (0, None),
        "SoNam_ViTriHienTai": (0, None),
        "SoNam_TuLanThangChuc": (0, None),
        "SoNam_VoiQuanLyHienTai": (0, None),
        "CapBacCongViec": (1, 5),
    }

    rows = []
    for col, (lo, hi) in rules.items():
        if col not in df.columns:
            continue
        series = pd.to_numeric(df[col], errors="coerce")
        invalid_mask = series.isna() & df[col].notna()
        if lo is not None:
            invalid_mask = invalid_mask | (series < lo)
        if hi is not None:
            invalid_mask = invalid_mask | (series > hi)
        n_invalid = int(invalid_mask.sum())
        rows.append(
            {
                "Column": col,
                "Lower Rule": lo,
                "Upper Rule": hi,
                "Invalid Count": n_invalid,
                "Invalid %": round(n_invalid / len(df) * 100, 2) if len(df) else 0.0,
            }
        )
    return pd.DataFrame(rows)


def validate_dataset(df: pd.DataFrame) -> dict[str, Any]:
    """Tổng hợp kết quả validation."""
    missing_df = check_missing_values(df)
    total_missing = int(missing_df["Missing Count"].sum())
    dup = check_duplicates(df)

    return {
        "missing_summary": missing_df,
        "total_missing": total_missing,
        "missing_message": (
            "Không phát hiện missing values."
            if total_missing == 0
            else f"Phát hiện tổng {total_missing} giá trị thiếu."
        ),
        "duplicates": dup,
        "dtypes": check_data_types(df),
        "categorical_uniques": get_categorical_uniques(df),
        "invalid_numeric": check_invalid_numeric(df),
        "numeric_columns_present": [c for c in NUMERIC_COLUMNS if c in df.columns],
    }
