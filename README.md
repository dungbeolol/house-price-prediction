# House Price Prediction

Dự đoán giá nhà ở khu vực Washington (King County) dựa trên các đặc trưng
như diện tích, số phòng, vị trí, năm xây dựng...

## Cấu trúc project

```
house-price-prediction/
├── config.yaml           # Mọi tham số: đường dẫn, seed, siêu tham số, stacking, tuning
├── data/
│   ├── raw/              # Dữ liệu gốc, không chỉnh sửa trực tiếp
│   └── processed/        # Dữ liệu sau khi làm sạch (sinh ra từ src/preprocessing.py)
├── notebooks/            # Notebook khám phá theo từng bước
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_modeling.ipynb
│   ├── 04_evaluation.ipynb
│   ├── 05_tuning.ipynb
│   ├── 06_stacking.ipynb   # Stacking: so với base model, trọng số trộn, đa dạng sai số
│   └── 07_inference.ipynb  # Dự đoán nhà mới, kiểm tra phiên bản thư viện, chạy test
├── src/                  # Code tái sử dụng, chạy được độc lập qua CLI
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── train.py
│   ├── evaluate.py
│   ├── tune.py
│   ├── predict.py        # Dự đoán giá cho căn nhà mới (chưa có trong data)
│   ├── config.py         # Đọc + kiểm tra config.yaml
│   └── utils.py
├── tests/                # pytest: config, preprocessing, predict, train/stacking, evaluate, utils
├── examples/              # File JSON mẫu để test predict.py
│   ├── sample_house.json
│   └── sample_houses.json
├── models/
│   ├── best_model.pkl          # Model tốt nhất đã train
│   ├── feature_schema.json     # Bộ cột chuẩn (sinh bởi preprocessing.py, dùng bởi predict.py)
│   └── model_meta.json         # Model nào, khi nào, phiên bản thư viện (sinh bởi train.py/tune.py)
├── outputs/
│   ├── figures/           # Biểu đồ xuất ra
│   └── reports/           # Bảng so sánh model, kết quả dạng CSV
├── requirements.txt        # Thư viện chạy (có cận trên/dưới)
├── requirements-dev.txt    # + pytest
├── pytest.ini
└── README.md
```

## Cách chạy

1. Cài thư viện:
   ```bash
   pip install -r requirements.txt          # chỉ để chạy
   pip install -r requirements-dev.txt      # + pytest, nếu muốn chạy test
   ```

   Đường dẫn trong `config.yaml` được hiểu tương đối so với **thư mục gốc
   project**, nên chạy script từ thư mục gốc hay từ `src/` đều được.

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

5. Hyperparameter tuning cho XGBoost/RandomForest/CatBoost bằng
   RandomizedSearchCV, tự động cập nhật `models/best_model.pkl` nếu tìm
   được tham số tốt hơn:
   ```bash
   python tune.py
   ```

6. Dự đoán giá cho 1 căn nhà mới (chưa có trong dữ liệu):
   ```bash
   python predict.py                            # chạy demo với nhà mẫu
   python predict.py --json ../examples/sample_house.json    # 1 căn nhà
   python predict.py --json ../examples/sample_houses.json   # nhiều căn nhà (list JSON)
   ```

   Hoặc mở từng notebook trong `notebooks/` để khám phá từng bước một cách trực quan.

### Notebook

Chạy theo thứ tự `01` → `07`. Tất cả đọc đường dẫn, seed, số fold và cấu hình
Stacking từ `config.yaml`, nên đổi config là notebook đổi theo.

- `03_modeling` và `05_tuning` **ghi đè** `models/best_model.pkl` và
  `models/model_meta.json` (giống `train.py` / `tune.py`). Sao lưu 2 file này
  trước nếu muốn giữ model hiện tại.
- `06_stacking` chỉ để phân tích, không ghi đè model đã lưu. Bước ablation
  (bỏ từng base model) mặc định tắt vì chậm; bật bằng `RUN_ABLATION = True`.
- Các bảng kết quả bằng chữ trong notebook 03 và 05 là số của 8 model đơn đã
  chạy trước đây; số của Stacking chỉ có sau khi bạn chạy trên dữ liệu thật.

### Chạy test

```bash
pytest                      # toàn bộ
pytest tests/test_predict.py -k consistency
```

Test dùng dữ liệu giả sinh ngay trong `tests/conftest.py` nên không cần
`data/` hay model đã train. Riêng `test_train.py` cần xgboost + catboost
(tự bỏ qua nếu chưa cài). Test quan trọng nhất là
`test_train_serve_consistency`: đảm bảo 1 căn nhà đi qua pipeline lúc
train và qua `predict.py` cho ra **đúng cùng vector feature**.

### Cấu hình (`config.yaml`)

Sửa tham số ở đây thay vì sửa hằng số trong code: seed, tỉ lệ split, số
fold, siêu tham số 8 model, cấu hình stacking, số vòng tuning, đường dẫn.
Muốn thử cấu hình khác mà không đụng file gốc:

```bash
HPP_CONFIG=config_thu_nghiem.yaml python src/train.py     # Linux/macOS
$env:HPP_CONFIG="config_thu_nghiem.yaml"; python src/train.py   # PowerShell
```

`config.py` kiểm tra config khi khởi động, thiếu mục nào sẽ báo lỗi ngay
kèm tên mục đó.

### Stacking ensemble

`train.py` hiện so sánh **9 model**: 8 model đơn + `Stacking`. Stacking huấn
luyện các base model (mặc định `RidgeCV`, `ExtraTrees`, `XGBoost`,
`CatBoost`), rồi một `RidgeCV` cấp 2 học cách trộn dự đoán ngoài-fold của
chúng. Chỉnh trong `config.yaml`, mục `stacking`:

- `enabled: false` để tắt (quay về đúng 8 model như trước).
- `base_models`: danh sách base model. Nên chọn các model **sai khác
  nhau** (tuyến tính + bagging + boosting); thêm model giống nhau chỉ tốn
  thời gian.
- **Chi phí**: mỗi lần fit, mỗi base model được train `cv_splits + 1` lần,
  và `train.py` còn bọc thêm 5-fold CV bên ngoài -> Stacking chậm hơn
  tổng thời gian train riêng các base model đó khoảng 6 lần. Nếu chỉ muốn thử nhanh, hạ `cv_splits` hoặc
  bớt base model.
- Khi Stacking là model tốt nhất, `evaluate.py` không vẽ feature
  importance (không có ý nghĩa cho stacking) mà in **trọng số trộn** của
  từng base model.

> Stacking chưa được chạy trên dữ liệu thật của bạn ở bản này, nên chưa có
> con số RMSLE của nó trong bảng kết quả bên dưới. Chạy `python
> src/train.py` để có số, rồi cập nhật bảng.

### Pin phiên bản

`requirements.txt` dùng cận dưới/cận trên (ví dụ `scikit-learn>=1.3,<2`)
để chặn nhảy major version. File `models/best_model.pkl` chỉ đọc lại an
toàn với đúng phiên bản thư viện lúc lưu, vì vậy:

- `train.py`/`tune.py` ghi `models/model_meta.json` (tên model, phiên bản
  scikit-learn/xgboost/catboost/pandas/numpy lúc lưu). `predict.py` so
  sánh và **cảnh báo** nếu môi trường hiện tại lệch.
- Để tái lập chính xác môi trường, sau khi mọi thứ chạy tốt hãy sinh file
  lock **trên máy bạn** và commit nó:
  ```bash
  pip freeze > requirements-lock.txt
  pip install -r requirements-lock.txt     # trên máy khác
  ```

### Dùng `predict.py` cho căn nhà của riêng bạn

Tạo 1 file JSON với đủ các trường sau (xem `examples/sample_house.json`):

```json
{
  "date": "2024-06-15",
  "bedrooms": 3, "bathrooms": 2, "sqft_living": 1800, "sqft_lot": 5000,
  "floors": 1, "waterfront": 0, "view": 0, "condition": 3,
  "sqft_above": 1800, "sqft_basement": 0,
  "yr_built": 1995, "yr_renovated": 0, "city": "Seattle"
}
```

Sau đó chạy:
```bash
python src/predict.py --json duong-dan-file-cua-ban.json
```

**Cách hoạt động:** `predict.py` áp lại đúng các bước feature engineering
đã dùng lúc train (`engineer_features()` trong `preprocessing.py`), rồi
dùng `models/feature_schema.json` (bộ cột + danh sách thành phố đã lưu
lúc train) để dựng đúng input model cần — kể cả khi:
- Thành phố trong input không nằm trong danh sách thành phố phổ biến lúc
  train → tự động gộp vào nhóm `"Other"` (không lỗi, không crash).
- Thiếu trường bắt buộc → báo lỗi rõ ràng thay vì âm thầm dự đoán sai.

## Bài toán

- **Input**: các đặc trưng của căn nhà (diện tích, số phòng, vị trí, năm xây...)
- **Output**: giá nhà dự đoán (`price`), hồi quy (regression)
- **Metric đánh giá chính**: RMSLE và R² tính trên **log-scale** (xem lý do bên dưới).
  Metric phụ để dễ hình dung bằng USD: MAE, MdAPE (median absolute % error).

## Kết quả hiện tại (Phần 4, 8 model)

| Model | CV RMSLE | Test RMSLE | Test R²(log) | Test MAE |
|---|---|---|---|---|
| LinearRegression | 0.302 | 0.308 | 0.676 | $159,585 |
| RidgeCV | 0.302 | 0.308 | 0.677 | $159,560 |
| DecisionTree | 0.358 | 0.362 | 0.553 | $150,143 |
| RandomForest | 0.313 | 0.315 | 0.661 | $125,640 |
| ExtraTrees | 0.310 | 0.301 | 0.691 | $119,911 |
| GradientBoosting | 0.302 | 0.303 | 0.686 | $116,450 |
| XGBoost | 0.296 | 0.296 | 0.701 | $112,725 |
| **CatBoost (best)** | **0.294** | **0.294** | **0.706** | $112,998 |

Nhóm gradient boosting (GradientBoosting, XGBoost, CatBoost) dẫn đầu,
đúng như kỳ vọng lý thuyết với dữ liệu dạng bảng.

## Kết quả sau Phần 6 (Tuning)

`src/tune.py` tune 3 model tiềm năng nhất (XGBoost, RandomForest,
CatBoost) bằng RandomizedSearchCV, so sánh công bằng bằng 5-fold CV:

| Model | RMSLE (baseline, Phần 4) | RMSLE (tuned) |
|---|---|---|
| XGBoost | 0.2958 | 0.2936 |
| RandomForest | 0.3133 | 0.3089 |
| **CatBoost** | 0.2941 | **0.2931** |

**Model cuối cùng: CatBoost đã tune** (RMSLE 5-fold = 0.2933), tự động
lưu vào `models/best_model.pkl`. Ở cả 3 model, tuning chỉ cải thiện nhẹ
(~0.3-1.4%) so với baseline — xem phần "bài học" bên dưới.

### Phân tích lỗi theo phân khúc giá (Phần 5, model CatBoost đã tune)

| Phân khúc | Khoảng giá | MdAPE | MAE (USD) |
|---|---|---|---|
| Q1 (rẻ nhất) | $84k - $300k | 16.7% | $60,519 |
| Q2 | $301k - $415k | 11.1% | $55,160 |
| Q3 | $417k - $539k | 11.0% | $66,160 |
| Q4 | $540k - $723k | 11.9% | $87,052 |
| Q5 (đắt nhất) | $725k - $4.67M | 12.4% | $189,047 |

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
- **Tuning không phải lúc nào cũng cải thiện nhiều**: sau khi tune kỹ cả
  3 model (XGBoost, RandomForest, CatBoost) bằng RandomizedSearchCV,
  RMSLE chỉ cải thiện 0.3-1.4% — cho thấy giới hạn hiện tại nằm ở
  **dữ liệu** (thiếu thông tin vị trí chi tiết) chứ không phải tham số
  model chưa tối ưu. Bài học: khi tuning không mang lại cải thiện đáng
  kể, nên đầu tư vào feature engineering/thu thập thêm dữ liệu thay vì
  tiếp tục tinh chỉnh tham số.
- **Model chỉ "dùng được" khi có cách dự đoán cho dữ liệu thực tế
  mới**: train/evaluate/tune xong không có nghĩa là xong — cần lưu lại
  đúng bộ cột (`feature_schema.json`) và danh sách thành phố đã gộp lúc
  train, để `predict.py` dựng lại chính xác input cho 1 căn nhà mới,
  kể cả khi nhà đó ở thành phố chưa từng xuất hiện lúc train (tự động
  gộp vào `"Other"` thay vì lỗi hoặc lệch cột).
- **CatBoost cần `bootstrap_type` phù hợp để tune `subsample`**: mặc
  định CatBoost dùng `bootstrap_type="MVS"`, không nhận tham số
  `subsample` — phải khai báo cố định `bootstrap_type="Bernoulli"` khi
  muốn đưa `subsample` vào không gian tìm kiếm.
