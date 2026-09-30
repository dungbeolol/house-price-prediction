"""
predict.py
Dự đoán giá cho 1 (hoặc nhiều) căn nhà mới bằng model đã train.

Cách dùng:
    python src/predict.py                      # chạy demo với nhà mẫu
    python src/predict.py --json house.json     # dự đoán 1 căn nhà
    python src/predict.py --json houses.json    # house.json là 1 list -> dự đoán nhiều căn

house.json cần có đủ các trường trong RAW_REQUIRED_FIELDS (xem
preprocessing.py), ví dụ:
{
  "date": "2024-06-15",
  "bedrooms": 3, "bathrooms": 2, "sqft_living": 1800, "sqft_lot": 5000,
  "floors": 1, "waterfront": 0, "view": 0, "condition": 3,
  "sqft_above": 1800, "sqft_basement": 0,
  "yr_built": 1995, "yr_renovated": 0, "city": "Seattle"
}
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

from config import CFG, resolve_path
from preprocessing import engineer_features, RAW_REQUIRED_FIELDS
from utils import warn_if_incompatible

MODEL_PATH = resolve_path(CFG["paths"]["model"])
SCHEMA_PATH = resolve_path(CFG["paths"]["schema"])
META_PATH = resolve_path(CFG["paths"]["model_meta"])


def load_artifacts():
    """Tải model + schema (bộ cột chuẩn được lưu lúc train).

    Nếu model.pkl và schema.json không khớp (ví dụ bạn train lại với
    tập cột khác nhưng quên chạy lại preprocessing.py), lỗi sẽ hiện rõ
    ràng ở bước reindex bên dưới thay vì âm thầm dự đoán sai."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Chưa có {MODEL_PATH}. Hãy chạy train.py trước.")
    if not SCHEMA_PATH.exists():
        raise FileNotFoundError(
            f"Chưa có {SCHEMA_PATH}. Hãy chạy preprocessing.py trước "
            "(hàm run_pipeline() sẽ tự sinh file schema này)."
        )
    # Cảnh báo (không chặn) nếu phiên bản thư viện lệch so với lúc lưu model
    if META_PATH.exists():
        warn_if_incompatible(json.loads(META_PATH.read_text()))
    model = joblib.load(MODEL_PATH)
    schema = json.loads(SCHEMA_PATH.read_text())
    return model, schema


def validate_raw_input(raw: dict) -> None:
    missing = [f for f in RAW_REQUIRED_FIELDS if f not in raw]
    if missing:
        raise ValueError(f"Thiếu các trường bắt buộc: {missing}")


def build_features(raw: dict, schema: dict) -> pd.DataFrame:
    """Biến 1 dict thông tin nhà thô thành đúng 1 dòng feature mà model
    kỳ vọng -- áp dụng lại y hệt logic feature engineering + encoding
    của preprocessing.py, rồi reindex về đúng bộ cột đã lưu trong schema
    (thêm cột thiếu = 0, bỏ cột thừa, đúng thứ tự)."""
    validate_raw_input(raw)
    df = pd.DataFrame([raw])

    df = engineer_features(df)

    # Gộp thành phố hiếm/lạ thành "Other" -- dùng ĐÚNG danh sách
    # kept_cities đã lưu lúc train, không tính lại từ dữ liệu mới (vì
    # dữ liệu mới chỉ có 1 dòng, không đủ để biết thành phố nào "hiếm").
    kept_cities = set(schema["kept_cities"])
    df["city"] = df["city"].where(df["city"].isin(kept_cities), "Other")
    df = pd.get_dummies(df, columns=["city"])  # không cần drop_first ở
    # đây: cột baseline (bị drop lúc train) sẽ tự động bị loại bỏ ở
    # bước reindex bên dưới vì nó không nằm trong schema["feature_columns"].

    df = df.reindex(columns=schema["feature_columns"], fill_value=0)
    return df


def predict_price(raw: dict, model=None, schema=None) -> float:
    """Dự đoán giá (USD) cho 1 căn nhà, từ 1 dict thông tin thô."""
    if model is None or schema is None:
        model, schema = load_artifacts()
    X = build_features(raw, schema)
    pred_log = model.predict(X)[0]
    return float(np.expm1(pred_log))


def predict_many(raw_list: list) -> list:
    model, schema = load_artifacts()
    return [predict_price(raw, model, schema) for raw in raw_list]


DEMO_HOUSE = {
    "date": "2024-06-15",
    "bedrooms": 3, "bathrooms": 2, "sqft_living": 1800, "sqft_lot": 5000,
    "floors": 1, "waterfront": 0, "view": 0, "condition": 3,
    "sqft_above": 1800, "sqft_basement": 0,
    "yr_built": 1995, "yr_renovated": 0, "city": "Seattle",
}


def main():
    parser = argparse.ArgumentParser(description="Dự đoán giá nhà")
    parser.add_argument(
        "--json", type=str, default=None,
        help="Đường dẫn file JSON: 1 object (1 căn nhà) hoặc 1 list (nhiều căn nhà)."
    )
    args = parser.parse_args()

    if args.json is None:
        print("Không có --json, chạy demo với 1 căn nhà mẫu:")
        print(json.dumps(DEMO_HOUSE, ensure_ascii=False, indent=2))
        price = predict_price(DEMO_HOUSE)
        print(f"\n=> Giá dự đoán: ${price:,.0f}")
        return

    try:
        data = json.loads(Path(args.json).read_text())

        if isinstance(data, list):
            prices = predict_many(data)
            for i, (raw, price) in enumerate(zip(data, prices)):
                label = raw.get("city", f"house_{i}")
                print(f"[{i}] {label}: ${price:,.0f}")
        else:
            price = predict_price(data)
            print(f"Giá dự đoán: ${price:,.0f}")
    except (ValueError, FileNotFoundError, KeyError) as e:
        print(f"Lỗi: {e}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
