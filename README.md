# PeopleRisk AI — HR Analytics Desktop

Ứng dụng **dataset-driven**: upload CSV → phân tích. Không kèm dataset mặc định.

## Cài đặt và chạy

### Windows (PowerShell)

1. Cài Git và Python, sau đó mở PowerShell tại thư mục muốn lưu dự án.
2. Tải mã nguồn và chuyển vào thư mục dự án:

```powershell
git clone https://github.com/vutheminh20222735/PhanTichNhanSuCD3.git
cd PhanTichNhanSuCD3
```

3. Tạo môi trường Python và cài các thư viện cần thiết:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r .\hr_analytics\requirements.txt
```

4. Khởi động ứng dụng:

```powershell
python .\hr_analytics\ung_dung.py
```

### Linux hoặc macOS

Từ thư mục chứa dự án đã tải về, tạo môi trường Python, cài các thư viện và chạy ứng dụng:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r hr_analytics/requirements.txt
bash hr_analytics/run_desktop.sh
```

Sau khi mở ứng dụng, chọn tệp CSV để bắt đầu phân tích. Dự án không kèm dataset mặc định.

## Cấu trúc (đơn giản)

```text
hr_analytics/
  ung_dung.py          # ứng dụng chính (CustomTkinter)
  run_desktop.sh
  giao_dien/           # giao diện
  xu_ly/               # đọc, làm sạch, EDA, tiện ích
  mo_hinh/             # train, đánh giá, dự đoán
  ket_qua/             # insight, khuyến nghị
  data/                # raw/processed (do người dùng upload)
  models/
```

| Thư mục | Việc chính |
|---------|------------|
| `xu_ly/` | Đọc CSV, profile, chất lượng, làm sạch, EDA |
| `mo_hinh/` | Phân loại / hồi quy / dự đoán |
| `ket_qua/` | Insight + khuyến nghị |
| `giao_dien/` | Theme, widget, biểu đồ |
