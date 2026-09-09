"""Phát hiện schema động cho dataset HR bất kỳ (VN/EN, có dấu, gạch dưới)."""

from __future__ import annotations

import unicodedata
from typing import Any

import pandas as pd

COLUMN_ALIASES: dict[str, list[str]] = {
    "target": [
        "NghiViec", "Nghỉ_việc", "Nghi_viec", "Attrition", "Left", "Exited",
        "Turnover", "EmployeeStatus", "Status", "Churn", "Resigned",
    ],
    "id": [
        "MaNhanVien", "Mã_nhân_viên", "Ma_nhan_vien", "EmployeeID",
        "EmployeeNumber", "EmpID", "Employee Number",
    ],
    "age": ["Tuoi", "Tuổi", "Age"],
    "gender": ["GioiTinh", "Giới_tính", "Gioi_tinh", "Gender", "Sex"],
    "marital": [
        "TinhTrangHonNhan", "Tình_trạng_hôn_nhân", "Tinh_trang_hon_nhan",
        "MaritalStatus", "Marriage",
    ],
    "location": ["DiaDiemLamViec", "Location", "City", "OfficeLocation"],
    "contract": ["LoaiHopDong", "ContractType", "EmploymentType"],
    "education": [
        "TrinhDoHocVan", "Trình_độ_học_vấn", "Trinh_do_hoc_van",
        "Education", "EducationLevel",
    ],
    "field": [
        "LinhVucDaoTao", "ChuyenNganhDaoTao", "Chuyên_ngành_đào_tạo",
        "Chuyen_nganh_dao_tao", "EducationField", "Major",
    ],
    "department": ["PhongBan", "Phòng_ban", "Phong_ban", "Department", "Dept"],
    "job_role": [
        "ViTriCongViec", "VaiTroCongViec", "Vai_trò_công_việc", "Vai_tro_cong_viec",
        "JobRole", "Position",
    ],
    "job_level": [
        "CapBacCongViec", "Cấp_bậc_công_việc", "Cap_bac_cong_viec",
        "JobLevel", "Grade",
    ],
    "travel": [
        "TanSuatCongTac", "Tần_suất_công_tác", "Tan_suat_cong_tac",
        "BusinessTravel", "Travel",
    ],
    "overtime": [
        "LamThemGio", "Làm_thêm_giờ", "Lam_them_gio", "OverTime", "Overtime", "OT",
    ],
    "income": [
        "ThuNhapHangThang_VND", "ThuNhapHangThang", "Thu_nhập_hàng_tháng",
        "Thu_nhap_hang_thang", "MonthlyIncome", "Salary", "Income", "ThuNhap", "Wage",
    ],
    "salary_hike": [
        "PhanTramTangLuong", "TyLeTangLuongPhanTram", "Tỷ_lệ_tăng_lương_phần_trăm",
        "Ty_le_tang_luong_phan_tram", "PercentSalaryHike", "SalaryHike",
    ],
    "distance": [
        "KhoangCachNha_Km", "KhoangCachTuNha", "Khoảng_cách_từ_nhà",
        "Khoang_cach_tu_nha", "DistanceFromHome", "Distance", "Commute",
    ],
    "job_satisfaction": [
        "HaiLong_CongViec", "MucHaiLongCongViec", "Mức_hài_lòng_công_việc",
        "Muc_hai_long_cong_viec", "JobSatisfaction",
    ],
    "env_satisfaction": [
        "HaiLong_MoiTruong", "MucHaiLongMoiTruong", "Mức_hài_lòng_môi_trường",
        "Muc_hai_long_moi_truong", "EnvironmentSatisfaction",
    ],
    "worklife": [
        "CanBangCongViec_CuocSong", "Cân_bằng_công_việc_cuộc_sống",
        "Can_bang_cong_viec_cuoc_song", "WorkLifeBalance", "WorkLife",
    ],
    "involvement": [
        "MucDoGanKetCongViec", "Mức_độ_gắn_kết_công_việc", "Muc_do_gan_ket_cong_viec",
        "JobInvolvement", "Engagement",
    ],
    "relationship": [
        "HaiLong_QuanHe", "MucHaiLongMoiQuanHe", "Mức_hài_lòng_mối_quan_hệ",
        "Muc_hai_long_moi_quan_he", "RelationshipSatisfaction",
    ],
    "performance": [
        "DanhGiaHieuSuat", "Đánh_giá_hiệu_suất", "Danh_gia_hieu_suat",
        "PerformanceRating", "Performance",
    ],
    "training": [
        "SoLanDaoTao_Nam", "SoLanDaoTaoNamTruoc", "Số_lần_đào_tạo_năm_trước",
        "So_lan_dao_tao_nam_truoc", "TrainingTimesLastYear",
    ],
    "num_companies": [
        "SoCongTyDaLam", "Số_công_ty_đã_làm", "So_cong_ty_da_lam", "NumCompaniesWorked",
    ],
    "total_experience": [
        "TongNamKinhNghiem", "TongSoNamLamViec", "Tổng_số_năm_làm_việc",
        "Tong_so_nam_lam_viec", "TotalWorkingYears", "Experience",
    ],
    "tenure": [
        "SoNamTaiCongTy", "Số_năm_tại_công_ty", "So_nam_tai_cong_ty",
        "YearsAtCompany", "Tenure",
    ],
    "years_role": [
        "SoNam_ViTriHienTai", "SoNamOVaiTroHienTai", "Số_năm_ở_vai_trò_hiện_tại",
        "So_nam_o_vai_tro_hien_tai", "YearsInCurrentRole",
    ],
    "years_promo": [
        "SoNam_TuLanThangChuc", "SoNamTuLanThangChucGanNhat",
        "Số_năm_từ_lần_thăng_chức_gần_nhất", "So_nam_tu_lan_thang_chuc_gan_nhat",
        "YearsSinceLastPromotion",
    ],
    "years_manager": [
        "SoNam_VoiQuanLyHienTai", "SoNamLamVoiQuanLyHienTai",
        "Số_năm_làm_với_quản_lý_hiện_tại", "So_nam_lam_voi_quan_ly_hien_tai",
        "YearsWithCurrManager",
    ],
    # Cột không dùng làm feature (hằng số / metadata HRIS)
    "junk": [
        "EmployeeCount", "SoLuongNhanVien", "Số_lượng_nhân_viên", "So_luong_nhan_vien",
        "StandardHours", "SoGioTieuChuan", "Số_giờ_tiêu_chuẩn", "So_gio_tieu_chuan",
        "Over18", "Tren18Tuoi", "Trên_18_tuổi", "Tren_18_tuoi",
    ],
}

# Alias ngắn dễ khớp nhầm — chỉ dùng exact match sau normalize
_SHORT_ALIASES = {
    "id", "ot", "sex", "role", "level", "grade", "dept", "wage", "left", "status",
}

POSITIVE_LEAVE_VALUES = {
    "có", "co", "yes", "y", "true", "1", "left", "exited",
    "attrition", "resigned", "nghỉ việc", "nghi viec", "nghỉ_việc", "nghi_viec",
}
NEGATIVE_STAY_VALUES = {
    "không", "khong", "no", "n", "false", "0", "stayed", "active",
    "ở lại", "o lai", "current",
}


def normalize_key(value: Any) -> str:
    """Chuẩn hóa tên cột/alias: bỏ dấu, khoảng trắng, gạch dưới → so khớp ổn định."""
    s = str(value).strip().lower()
    s = s.replace("đ", "d").replace("Đ", "d")
    s = unicodedata.normalize("NFD", s)
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    for ch in " _-./()[]{}":
        s = s.replace(ch, "")
    return s


def find_column(df: pd.DataFrame, aliases: list[str] | str, role: str | None = None) -> str | None:
    """Tìm cột theo alias (không phân biệt hoa thường / dấu / khoảng trắng)."""
    if isinstance(aliases, str):
        aliases = COLUMN_ALIASES.get(aliases, [aliases])
    elif role is not None:
        aliases = COLUMN_ALIASES.get(role, aliases)

    normalized = {normalize_key(c): c for c in df.columns}

    # 1) Exact sau normalize
    for alias in aliases:
        key = normalize_key(alias)
        if key in normalized:
            return normalized[key]

    # 2) Partial: alias đủ dài, ưu tiên khớp dài nhất
    candidates: list[tuple[int, str]] = []
    for alias in aliases:
        key = normalize_key(alias)
        if len(key) < 5 or key in _SHORT_ALIASES:
            continue
        for nk, original in normalized.items():
            if len(nk) < 4:
                continue
            if key == nk:
                candidates.append((10_000 + len(key), original))
            elif key in nk or nk in key:
                # tránh khớp quá lỏng (vd. "age" trong nhiều chuỗi)
                score = min(len(key), len(nk))
                if abs(len(key) - len(nk)) > max(len(key), len(nk)) * 0.6:
                    continue
                candidates.append((score, original))
    if candidates:
        candidates.sort(key=lambda x: (-x[0], x[1]))
        return candidates[0][1]
    return None


def find_role(df: pd.DataFrame, role: str) -> str | None:
    return find_column(df, COLUMN_ALIASES.get(role, []), role=role)


def detect_junk_columns(df: pd.DataFrame) -> list[str]:
    """Cột không nên đưa vào model: junk alias + hằng số (1 giá trị)."""
    found: list[str] = []
    for alias in COLUMN_ALIASES.get("junk", []):
        col = find_column(df, [alias])
        if col and col not in found:
            found.append(col)
    for col in df.columns:
        if col in found:
            continue
        if df[col].nunique(dropna=True) <= 1:
            found.append(col)
    return found


def detect_target_candidates(df: pd.DataFrame) -> list[str]:
    """Ứng viên target: alias + cột binary/low-cardinality."""
    found: list[str] = []
    primary = find_role(df, "target")
    if primary:
        found.append(primary)

    for col in df.columns:
        if col in found:
            continue
        nunique = df[col].nunique(dropna=True)
        if nunique == 2:
            vals = {str(v).strip().lower() for v in df[col].dropna().unique()}
            if vals & POSITIVE_LEAVE_VALUES or vals & NEGATIVE_STAY_VALUES:
                found.append(col)
            elif normalize_key(col) in {normalize_key(a) for a in COLUMN_ALIASES["target"]}:
                found.append(col)
    for alias in COLUMN_ALIASES["target"]:
        col = find_column(df, [alias])
        if col and col not in found:
            found.append(col)
    return found


def classify_columns(
    df: pd.DataFrame,
    target: str | None = None,
    id_col: str | None = None,
    junk_cols: list[str] | None = None,
) -> dict[str, list[str]]:
    """Phân loại numeric / categorical, loại id, target, junk."""
    exclude = {c for c in [target, id_col, *(junk_cols or [])] if c}
    numeric: list[str] = []
    categorical: list[str] = []
    for col in df.columns:
        if col in exclude:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            numeric.append(col)
        else:
            categorical.append(col)
    return {"numeric": numeric, "categorical": categorical}


def map_binary_target(y: pd.Series) -> pd.Series:
    """Map target binary linh hoạt → 0 stay / 1 leave."""
    def _map(v: Any) -> int | float:
        if pd.isna(v):
            return float("nan")
        if isinstance(v, (int, float)) and v in (0, 1):
            return int(v)
        s = str(v).strip().lower()
        if s in POSITIVE_LEAVE_VALUES:
            return 1
        if s in NEGATIVE_STAY_VALUES:
            return 0
        return float("nan")

    mapped = y.map(_map)
    if mapped.isna().any():
        uniques = [u for u in y.dropna().unique()]
        if len(uniques) == 2:
            counts = y.value_counts()
            leave_label = counts.idxmin()
            mapped = y.map(lambda v: 1 if v == leave_label else 0)
        else:
            bad = y[mapped.isna()].unique().tolist()
            raise ValueError(f"Không map được target: {bad}")
    return mapped.astype(int)


def leave_label_from_target(series: pd.Series) -> Any:
    """Giá trị gốc tương ứng lớp nghỉ việc (encoded=1)."""
    mapped = map_binary_target(series)
    for raw, enc in zip(series, mapped):
        if enc == 1:
            return raw
    return series.iloc[0]


def build_schema(df: pd.DataFrame, target: str | None = None) -> dict[str, Any]:
    """Tạo schema profile đầy đủ cho UI."""
    id_col = find_role(df, "id")
    junk_cols = detect_junk_columns(df)
    # ID cũng coi là junk nếu phát hiện
    if id_col and id_col not in junk_cols:
        junk_cols = [id_col, *junk_cols]

    candidates = detect_target_candidates(df)
    target_col = target if target in df.columns else (candidates[0] if candidates else None)
    types = classify_columns(df, target=target_col, id_col=id_col, junk_cols=junk_cols)

    roles: dict[str, str | None] = {
        "id": id_col,
        "target": target_col,
        "department": find_role(df, "department"),
        "job_role": find_role(df, "job_role"),
        "overtime": find_role(df, "overtime"),
        "income": find_role(df, "income"),
        "age": find_role(df, "age"),
        "tenure": find_role(df, "tenure"),
        "job_satisfaction": find_role(df, "job_satisfaction"),
        "distance": find_role(df, "distance"),
        "performance": find_role(df, "performance"),
        "training": find_role(df, "training"),
        "gender": find_role(df, "gender"),
        "location": find_role(df, "location"),
        "contract": find_role(df, "contract"),
        "job_level": find_role(df, "job_level"),
        "worklife": find_role(df, "worklife"),
        "relationship": find_role(df, "relationship"),
        "env_satisfaction": find_role(df, "env_satisfaction"),
        "involvement": find_role(df, "involvement"),
        "salary_hike": find_role(df, "salary_hike"),
        "total_experience": find_role(df, "total_experience"),
        "num_companies": find_role(df, "num_companies"),
        "marital": find_role(df, "marital"),
        "education": find_role(df, "education"),
        "field": find_role(df, "field"),
        "travel": find_role(df, "travel"),
        "years_role": find_role(df, "years_role"),
        "years_promo": find_role(df, "years_promo"),
        "years_manager": find_role(df, "years_manager"),
    }

    filter_roles = ["department", "location", "gender", "overtime"]
    filters = [roles[r] for r in filter_roles if roles.get(r)]

    return {
        "n_rows": len(df),
        "n_cols": df.shape[1],
        "memory_mb": float(df.memory_usage(deep=True).sum() / (1024**2)),
        "numeric": types["numeric"],
        "categorical": types["categorical"],
        "n_numeric": len(types["numeric"]),
        "n_categorical": len(types["categorical"]),
        "id_col": id_col,
        "junk_cols": junk_cols,
        "target": target_col,
        "target_candidates": candidates,
        "roles": roles,
        "filter_columns": filters,
        "missing_total": int(df.isnull().sum().sum()),
        "duplicate_total": int(df.duplicated().sum()),
    }
