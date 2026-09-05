"""Tiện ích dùng chung cho HR Analytics."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "dataset_nhan_su.csv"
MODELS_DIR = PROJECT_ROOT / "models" / "saved_models"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

TARGET_COL = "NghiViec"
ID_COL = "MaNhanVien"
LEAVE_LABEL = "Có"
STAY_LABEL = "Không"

EXPECTED_COLUMNS = [
    "MaNhanVien",
    "Tuoi",
    "GioiTinh",
    "TinhTrangHonNhan",
    "DiaDiemLamViec",
    "LoaiHopDong",
    "TrinhDoHocVan",
    "LinhVucDaoTao",
    "PhongBan",
    "ViTriCongViec",
    "CapBacCongViec",
    "TanSuatCongTac",
    "LamThemGio",
    "ThuNhapHangThang_VND",
    "PhanTramTangLuong",
    "KhoangCachNha_Km",
    "HaiLong_CongViec",
    "HaiLong_MoiTruong",
    "CanBangCongViec_CuocSong",
    "MucDoGanKetCongViec",
    "HaiLong_QuanHe",
    "DanhGiaHieuSuat",
    "SoLanDaoTao_Nam",
    "SoCongTyDaLam",
    "TongNamKinhNghiem",
    "SoNamTaiCongTy",
    "SoNam_ViTriHienTai",
    "SoNam_TuLanThangChuc",
    "SoNam_VoiQuanLyHienTai",
    "NghiViec",
]

NUMERIC_COLUMNS = [
    "Tuoi",
    "CapBacCongViec",
    "ThuNhapHangThang_VND",
    "PhanTramTangLuong",
    "KhoangCachNha_Km",
    "HaiLong_CongViec",
    "HaiLong_MoiTruong",
    "CanBangCongViec_CuocSong",
    "MucDoGanKetCongViec",
    "HaiLong_QuanHe",
    "DanhGiaHieuSuat",
    "SoLanDaoTao_Nam",
    "SoCongTyDaLam",
    "TongNamKinhNghiem",
    "SoNamTaiCongTy",
    "SoNam_ViTriHienTai",
    "SoNam_TuLanThangChuc",
    "SoNam_VoiQuanLyHienTai",
]

CATEGORICAL_COLUMNS = [
    "GioiTinh",
    "TinhTrangHonNhan",
    "DiaDiemLamViec",
    "LoaiHopDong",
    "TrinhDoHocVan",
    "LinhVucDaoTao",
    "PhongBan",
    "ViTriCongViec",
    "TanSuatCongTac",
    "LamThemGio",
]

OUTLIER_COLUMNS = [
    "Tuoi",
    "ThuNhapHangThang_VND",
    "KhoangCachNha_Km",
    "TongNamKinhNghiem",
    "SoNamTaiCongTy",
]

DESCRIPTIVE_COLUMNS = [
    "Tuoi",
    "ThuNhapHangThang_VND",
    "KhoangCachNha_Km",
    "HaiLong_CongViec",
    "HaiLong_MoiTruong",
    "CanBangCongViec_CuocSong",
    "MucDoGanKetCongViec",
    "HaiLong_QuanHe",
    "DanhGiaHieuSuat",
    "SoLanDaoTao_Nam",
    "SoCongTyDaLam",
    "TongNamKinhNghiem",
    "SoNamTaiCongTy",
    "SoNam_ViTriHienTai",
    "SoNam_TuLanThangChuc",
    "SoNam_VoiQuanLyHienTai",
]

COLUMN_MEANINGS: dict[str, str] = {
    "MaNhanVien": "Mã định danh nhân viên",
    "Tuoi": "Tuổi của nhân viên",
    "GioiTinh": "Giới tính",
    "TinhTrangHonNhan": "Tình trạng hôn nhân",
    "DiaDiemLamViec": "Địa điểm làm việc",
    "LoaiHopDong": "Loại hợp đồng lao động",
    "TrinhDoHocVan": "Trình độ học vấn cao nhất",
    "LinhVucDaoTao": "Lĩnh vực đào tạo",
    "PhongBan": "Phòng ban đang công tác",
    "ViTriCongViec": "Vị trí / chức danh công việc",
    "CapBacCongViec": "Cấp bậc công việc (1–5)",
    "TanSuatCongTac": "Tần suất đi công tác",
    "LamThemGio": "Có làm thêm giờ hay không",
    "ThuNhapHangThang_VND": "Thu nhập hàng tháng (VND)",
    "PhanTramTangLuong": "Phần trăm tăng lương gần nhất",
    "KhoangCachNha_Km": "Khoảng cách từ nhà đến nơi làm việc (km)",
    "HaiLong_CongViec": "Mức hài lòng với công việc (1–5)",
    "HaiLong_MoiTruong": "Mức hài lòng với môi trường làm việc (1–5)",
    "CanBangCongViec_CuocSong": "Cân bằng công việc – cuộc sống (1–5)",
    "MucDoGanKetCongViec": "Mức độ gắn kết với công việc (1–5)",
    "HaiLong_QuanHe": "Hài lòng với quan hệ đồng nghiệp/quản lý (1–5)",
    "DanhGiaHieuSuat": "Đánh giá hiệu suất (1–5)",
    "SoLanDaoTao_Nam": "Số lần đào tạo trong năm",
    "SoCongTyDaLam": "Số công ty đã từng làm",
    "TongNamKinhNghiem": "Tổng số năm kinh nghiệm",
    "SoNamTaiCongTy": "Số năm làm việc tại công ty hiện tại",
    "SoNam_ViTriHienTai": "Số năm ở vị trí hiện tại",
    "SoNam_TuLanThangChuc": "Số năm kể từ lần thăng chức gần nhất",
    "SoNam_VoiQuanLyHienTai": "Số năm làm việc với quản lý hiện tại",
    "NghiViec": "Tình trạng nghỉ việc (Có / Không) — biến mục tiêu",
}

FILTER_COLUMNS = [
    "PhongBan",
    "DiaDiemLamViec",
    "GioiTinh",
    "LoaiHopDong",
    "CapBacCongViec",
    "LamThemGio",
]


def format_vnd(value: float | int | None) -> str:
    """Định dạng số tiền VND."""
    if value is None or pd.isna(value):
        return "N/A"
    return f"{int(round(float(value))):,} VND".replace(",", ".")


def format_vnd_short(value: float | int | None) -> str:
    """Định dạng tiền dạng triệu VND."""
    if value is None or pd.isna(value):
        return "N/A"
    return f"{float(value) / 1_000_000:.1f} triệu VND"


def format_vnd_compact(value: float | int | None) -> str:
    """Định dạng gọn cho KPI: 20.26M ₫."""
    if value is None or pd.isna(value):
        return "N/A"
    return f"{float(value) / 1_000_000:.2f}M ₫"


def format_pct(value: float | None, digits: int = 2) -> str:
    """Định dạng tỷ lệ phần trăm từ giá trị 0–1 hoặc đã là %."""
    if value is None or pd.isna(value):
        return "N/A"
    return f"{float(value):.{digits}f}%"


def attrition_mask(series: pd.Series) -> pd.Series:
    """Mask nhân viên nghỉ việc."""
    return series.astype(str).str.strip() == LEAVE_LABEL


def attrition_rate(df: pd.DataFrame, target: str = TARGET_COL) -> float:
    """Tỷ lệ nghỉ việc (%) trên dataframe."""
    if df.empty or target not in df.columns:
        return 0.0
    return float(attrition_mask(df[target]).mean() * 100)


def safe_div(numerator: float, denominator: float) -> float:
    """Chia an toàn."""
    if denominator == 0:
        return 0.0
    return numerator / denominator


def apply_filters(df: pd.DataFrame, filters: dict[str, Any]) -> pd.DataFrame:
    """Áp dụng bộ lọc lên dataframe.

    So khớp bằng chuỗi để tránh lệch dtype (int64 vs str, pandas StringDtype, …).
    """
    out = df
    for col, selected in filters.items():
        if col not in out.columns or selected is None:
            continue
        if isinstance(selected, (list, tuple, set)):
            if len(selected) == 0:
                continue
            want = {str(x) for x in selected}
            out = out[out[col].astype(str).isin(want)]
        else:
            out = out[out[col].astype(str) == str(selected)]
    if out is df:
        return df.copy()
    return out.reset_index(drop=True)
