# PeopleRisk AI — HR Analytics Desktop

Ứng dụng **dataset-driven**: upload CSV → phân tích. Không kèm dataset mặc định.

## Chạy

```bash
cd hr_analytics
pip install -r requirements.txt
./run_desktop.sh
```

Entry: `ung_dung.py`

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
