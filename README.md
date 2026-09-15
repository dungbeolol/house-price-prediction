# House Price Prediction

Dự đoán giá nhà ở khu vực Washington (King County) dựa trên các đặc trưng
như diện tích, số phòng, vị trí, năm xây dựng...

## Cấu trúc project

```
house-price-prediction/
├── data/
│   ├── raw/              # Dữ liệu gốc, không chỉnh sửa trực tiếp
│   └── processed/        # Dữ liệu sau khi làm sạch (sinh ra từ src/preprocessing.py)
├── notebooks/            # Notebook khám phá theo từng bước
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_modeling.ipynb
│   └── 04_evaluation.ipynb
├── src/                  # Code tái sử dụng, chạy được độc lập qua CLI
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── train.py
│   ├── evaluate.py
│   └── utils.py
├── models/                # Model đã train (.pkl)
├── outputs/
│   ├── figures/           # Biểu đồ xuất ra
│   └── reports/           # Bảng so sánh model, kết quả dạng CSV
├── requirements.txt
└── README.md
```

## Cách chạy

1. Cài thư viện:
   ```bash
   pip install -r requirements.txt
   ```

2. Chạy pipeline tiền xử lý (sinh ra `data/processed/cleaned_data.csv`):
   ```bash
   cd src
   python preprocessing.py
   ```

3. Train và so sánh các model (kết quả lưu vào `models/` và `outputs/reports/`):
   ```bash
   python train.py
   ```

4. Đánh giá chi tiết model tốt nhất (biểu đồ lưu vào `outputs/figures/`):
   ```bash
   python evaluate.py
   ```

   Hoặc mở từng notebook trong `notebooks/` để khám phá từng bước một cách trực quan.

## Bài toán

- **Input**: các đặc trưng của căn nhà (diện tích, số phòng, vị trí, năm xây...)
- **Output**: giá nhà dự đoán (`price`), hồi quy (regression)
- **Metric đánh giá**: RMSE, MAE, R²

## Ghi chú xử lý dữ liệu

- Cột `price_per_sqft` bị loại bỏ vì gây data leakage (được tính trực tiếp từ `price`).
- Các dòng có `price = 0` được loại bỏ (lỗi nhập liệu).
- Target `price` được log-transform (`log1p`) trước khi train vì phân phối lệch phải mạnh.
