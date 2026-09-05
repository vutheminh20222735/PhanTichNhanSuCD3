"""Phát hiện schema động cho dataset HR bất kỳ."""

from __future__ import annotations

from typing import Any

import pandas as pd

COLUMN_ALIASES: dict[str, list[str]] = {
    "target": [
        "NghiViec", "Attrition", "Left", "Exited", "Turnover",
        "EmployeeStatus", "Status", "Churn", "Resigned",
    ],
    "id": ["MaNhanVien", "EmployeeID", "EmployeeNumber", "EmpID", "ID", "Id"],
    "age": ["Tuoi", "Age"],
    "gender": ["GioiTinh", "Gender", "Sex"],
    "marital": ["TinhTrangHonNhan", "MaritalStatus", "Marriage"],
    "location": ["DiaDiemLamViec", "Location", "City", "OfficeLocation"],
    "contract": ["LoaiHopDong", "ContractType", "EmploymentType"],
    "education": ["TrinhDoHocVan", "Education", "EducationLevel"],
    "field": ["LinhVucDaoTao", "EducationField", "Major"],
    "department": ["PhongBan", "Department", "Dept"],
    "job_role": ["ViTriCongViec", "JobRole", "Role", "Position"],
    "job_level": ["CapBacCongViec", "JobLevel", "Level", "Grade"],
    "travel": ["TanSuatCongTac", "BusinessTravel", "Travel"],
    "overtime": ["LamThemGio", "OverTime", "Overtime", "OT"],
    "income": [
        "ThuNhapHangThang_VND", "MonthlyIncome", "Salary", "Income",
        "ThuNhap", "ThuNhapHangThang", "Wage",
    ],
    "salary_hike": ["PhanTramTangLuong", "PercentSalaryHike", "SalaryHike"],
    "distance": ["KhoangCachNha_Km", "DistanceFromHome", "Distance", "Commute"],
    "job_satisfaction": ["HaiLong_CongViec", "JobSatisfaction", "Satisfaction"],
    "env_satisfaction": ["HaiLong_MoiTruong", "EnvironmentSatisfaction"],
    "worklife": ["CanBangCongViec_CuocSong", "WorkLifeBalance", "WorkLife"],
    "involvement": ["MucDoGanKetCongViec", "JobInvolvement", "Engagement"],
    "relationship": ["HaiLong_QuanHe", "RelationshipSatisfaction"],
    "performance": ["DanhGiaHieuSuat", "PerformanceRating", "Performance"],
    "training": ["SoLanDaoTao_Nam", "TrainingTimesLastYear", "Training"],
    "num_companies": ["SoCongTyDaLam", "NumCompaniesWorked"],
    "total_experience": ["TongNamKinhNghiem", "TotalWorkingYears", "Experience"],
    "tenure": ["SoNamTaiCongTy", "YearsAtCompany", "Tenure"],
    "years_role": ["SoNam_ViTriHienTai", "YearsInCurrentRole"],
    "years_promo": ["SoNam_TuLanThangChuc", "YearsSinceLastPromotion"],
    "years_manager": ["SoNam_VoiQuanLyHienTai", "YearsWithCurrManager"],
}

POSITIVE_LEAVE_VALUES = {
    "có", "co", "yes", "y", "true", "1", "left", "exited",
    "attrition", "resigned", "nghỉ việc", "nghi viec",
}
NEGATIVE_STAY_VALUES = {
    "không", "khong", "no", "n", "false", "0", "stayed", "active",
    "ở lại", "o lai", "current",
}


def find_column(df: pd.DataFrame, aliases: list[str] | str, role: str | None = None) -> str | None:
    """Tìm cột theo alias (không phân biệt hoa thường / khoảng trắng)."""
    if isinstance(aliases, str):
        aliases = COLUMN_ALIASES.get(aliases, [aliases])
    elif role is not None:
        aliases = COLUMN_ALIASES.get(role, aliases)

    normalized = {str(c).strip().lower().replace(" ", ""): c for c in df.columns}
    for alias in aliases:
        key = str(alias).strip().lower().replace(" ", "")
        if key in normalized:
            return normalized[key]
    # partial contains
    for alias in aliases:
        key = str(alias).strip().lower().replace(" ", "")
        for nk, original in normalized.items():
            if key and (key in nk or nk in key):
                return original
    return None


def find_role(df: pd.DataFrame, role: str) -> str | None:
    return find_column(df, COLUMN_ALIASES.get(role, []), role=role)


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
            elif col.lower() in {a.lower() for a in COLUMN_ALIASES["target"]}:
                found.append(col)
    # alias matches not yet added
    for alias in COLUMN_ALIASES["target"]:
        col = find_column(df, [alias])
        if col and col not in found:
            found.append(col)
    return found


def classify_columns(df: pd.DataFrame, target: str | None = None, id_col: str | None = None) -> dict[str, list[str]]:
    """Phân loại numeric / categorical, loại id & target."""
    exclude = {c for c in [target, id_col] if c}
    numeric: list[str] = []
    categorical: list[str] = []
    for col in df.columns:
        if col in exclude:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            # low-cardinality numeric may still be numeric feature
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
        # fallback: minority class as leave if exactly 2 classes handled outside
        return float("nan")

    mapped = y.map(_map)
    if mapped.isna().any():
        # nếu còn NaN: dùng class ít hơn làm leave khi đúng 2 lớp
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
    # lấy một giá trị gốc có mapped=1
    for raw, enc in zip(series, mapped):
        if enc == 1:
            return raw
    return series.iloc[0]


def build_schema(df: pd.DataFrame, target: str | None = None) -> dict[str, Any]:
    """Tạo schema profile đầy đủ cho UI."""
    id_col = find_role(df, "id")
    candidates = detect_target_candidates(df)
    target_col = target if target in df.columns else (candidates[0] if candidates else None)
    types = classify_columns(df, target=target_col, id_col=id_col)

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
        "target": target_col,
        "target_candidates": candidates,
        "roles": roles,
        "filter_columns": filters,
        "missing_total": int(df.isnull().sum().sum()),
        "duplicate_total": int(df.duplicated().sum()),
    }
