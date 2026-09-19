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
│   ├── 04_evaluation.ipynb
│   └── 05_tuning.ipynb
├── src/                  # Code tái sử dụng, chạy được độc lập qua CLI
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── train.py
│   ├── evaluate.py
│   ├── tune.py
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

3. Train và so sánh các model bằng CV + tập test giữ riêng (lưu vào
   `models/best_model.pkl` và `outputs/reports/`):
   ```bash
   python train.py
   ```

4. Đánh giá chi tiết model tốt nhất — residual, sai số theo phân khúc
   giá, top dự đoán tệ nhất (biểu đồ + báo cáo lưu vào `outputs/`):
   ```bash
   python evaluate.py
   ```

5. Hyperparameter tuning cho XGBoost/RandomForest bằng RandomizedSearchCV,
   tự động cập nhật `models/best_model.pkl` nếu tìm được tham số tốt hơn:
   ```bash
   python tune.py
   ```

   Hoặc mở từng notebook trong `notebooks/` để khám phá từng bước một cách trực quan.

## Bài toán

- **Input**: các đặc trưng của căn nhà (diện tích, số phòng, vị trí, năm xây...)
- **Output**: giá nhà dự đoán (`price`), hồi quy (regression)
- **Metric đánh giá chính**: RMSLE và R² tính trên **log-scale** (xem lý do bên dưới).
  Metric phụ để dễ hình dung bằng USD: MAE, MdAPE (median absolute % error).

## Kết quả hiện tại (sau Phần 6 - Tuning)

| Model | CV RMSLE (baseline) | CV RMSLE (tuned) |
|---|---|---|
| XGBoost | 0.2958 | **0.2952** (best_model.pkl) |
| RandomForest | 0.3133 | 0.3081 |

Model cuối cùng (`models/best_model.pkl`) là XGBoost đã tune, với sai số
trung vị (MdAPE) khoảng 13%. Tuning chỉ cải thiện nhẹ so với baseline —
xem phần "bài học" bên dưới để hiểu vì sao.

### Phân tích lỗi theo phân khúc giá (Phần 5)

| Phân khúc | Khoảng giá | MdAPE | MAE (USD) |
|---|---|---|---|
| Q1 (rẻ nhất) | $84k - $300k | 15.4% | $60,990 |
| Q2 | $301k - $415k | 10.3% | $57,385 |
| Q3 | $417k - $539k | 11.0% | $66,263 |
| Q4 | $540k - $723k | 11.6% | $86,645 |
| Q5 (đắt nhất) | $725k - $4.67M | 13.0% | $195,568 |

Model dự đoán tốt nhất ở phân khúc giá trung bình (Q2-Q4), kém hơn một
chút ở 2 đầu (nhà rất rẻ hoặc rất đắt) — hợp lý vì các phân khúc này ít
mẫu hơn và đa dạng hơn.

## Bài học quan trọng rút ra trong quá trình làm

- **Data leakage**: cột `price_per_sqft` bị loại bỏ vì được tính trực tiếp từ `price`.
- **Dữ liệu lỗi**: các dòng có `price = 0` bị loại bỏ (49 dòng).
- **Log-transform target**: `price` được biến đổi bằng `log1p` trước khi train vì
  phân phối lệch phải mạnh (vài căn nhà giá hàng chục triệu USD).
- **Encoding city/statezip**: ban đầu one-hot cả `city` và `statezip` tạo
  ra 138 cột, khiến Linear/Ridge Regression **overfit nặng** (R² âm trên
  test) vì hai cột này gần như trùng thông tin (đa cộng tuyến) và nhiều
  category chỉ có vài mẫu. Đã sửa bằng cách: bỏ `statezip`, gộp các
  thành phố hiếm (<30 mẫu) thành nhóm `Other` trước khi one-hot city.
- **Chọn sai metric để đánh giá**: sau khi log-transform target, nếu tính
  RMSE/R² trên **thang giá USD gốc** (bằng cách `expm1` dự đoán rồi so
  sánh), một vài căn nhà siêu đắt bị lệch nhẹ trong log-space sẽ bị
  khuếch đại theo cấp số nhân thành sai số hàng chục triệu USD, khiến R²
  âm sâu dù model thực ra dự đoán khá tốt. Cách sửa: đánh giá bằng
  RMSLE/R² trên log-scale (ổn định), chỉ dùng MAE/MdAPE ở thang USD để
  tham khảo trực quan.
- **Residual trên USD gây hiểu lầm**: cùng lý do log-transform, residual
  tính trên USD gốc luôn trông như "phễu" (lớn dần theo giá) dù model
  không thực sự tệ hơn. Nên phân tích residual trên log-scale.
- **Tuning không phải lúc nào cũng cải thiện nhiều**: sau khi tune kỹ
  XGBoost/RandomForest bằng RandomizedSearchCV, RMSLE chỉ cải thiện
  0.2-2% — cho thấy giới hạn hiện tại nằm ở **dữ liệu** (thiếu thông tin
  vị trí chi tiết) chứ không phải tham số model chưa tối ưu. Bài học:
  khi tuning không mang lại cải thiện đáng kể, nên đầu tư vào feature
  engineering/thu thập thêm dữ liệu thay vì tiếp tục tinh chỉnh tham số.
