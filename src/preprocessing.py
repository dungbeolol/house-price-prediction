"""
preprocessing.py
Làm sạch dữ liệu + tạo feature mới cho bài toán dự đoán giá nhà.
Chạy trực tiếp: python src/preprocessing.py
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path

from config import CFG, resolve_path
from data_loader import load_raw_data, PROCESSED_DATA_PATH

SCHEMA_PATH = resolve_path(CFG["paths"]["schema"])
MIN_CITY_COUNT = CFG["preprocessing"]["min_city_count"]  # ngưỡng gộp thành phố hiếm thành "Other"

# Các trường thô bắt buộc phải có để dự đoán 1 căn nhà mới (predict.py)
RAW_REQUIRED_FIELDS = [
    "date", "bedrooms", "bathrooms", "sqft_living", "sqft_lot", "floors",
    "waterfront", "view", "condition", "sqft_above", "sqft_basement",
    "yr_built", "yr_renovated", "city",
]


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Loại bỏ dữ liệu lỗi/leakage."""
    df = df.copy()

    # Bỏ dòng price = 0 (lỗi nhập liệu)
    df = df[df["price"] > 0]

    # Bỏ cột leakage: price_per_sqft = price / sqft_living
    if "price_per_sqft" in df.columns:
        df = df.drop(columns=["price_per_sqft"])

    # Cột street quá chi tiết, không hữu ích cho model tổng quát
    if "street" in df.columns:
        df = df.drop(columns=["street"])

    return df.reset_index(drop=True)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Tạo các feature mới từ dữ liệu gốc."""
    df = df.copy()

    df["date"] = pd.to_datetime(df["date"])
    df["year_sold"] = df["date"].dt.year
    df["month_sold"] = df["date"].dt.month

    df["house_age"] = df["year_sold"] - df["yr_built"]
    df["was_renovated"] = (df["yr_renovated"] > 0).astype(int)
    df["years_since_renovation"] = np.where(
        df["was_renovated"] == 1,
        df["year_sold"] - df["yr_renovated"],
        df["house_age"],
    )
    df["total_rooms"] = df["bedrooms"] + df["bathrooms"]

    df = df.drop(columns=["date"])
    return df


def get_kept_cities(df: pd.DataFrame, min_count: int = MIN_CITY_COUNT) -> list:
    """Danh sách thành phố CÓ ĐỦ mẫu (>= min_count), không bị gộp vào
    "Other". Tách thành hàm riêng để `predict.py` tái sử dụng CHÍNH XÁC
    cùng 1 logic khi xử lý 1 căn nhà mới -- nếu không, thành phố hiếm ở
    predict có thể vô tình được one-hot thành cột model chưa từng thấy,
    hoặc ngược lại, một thành phố phổ biến bị nhét nhầm vào "Other"."""
    city_counts = df["city"].value_counts()
    return sorted(city_counts[city_counts >= min_count].index.tolist())


def encode_categorical(df: pd.DataFrame, min_count: int = MIN_CITY_COUNT) -> pd.DataFrame:
    """One-hot encode cột city (đã gộp nhóm hiếm) và loại statezip.

    Lý do:
    - `statezip` gần như trùng thông tin với `city` (mỗi statezip nằm
      trong đúng 1 city) -> giữ cả 2 gây đa cộng tuyến và bùng nổ số
      chiều (one-hot 2 cột x hàng chục category -> hàng trăm cột),
      khiến Linear/Ridge Regression overfit nặng (R2 âm khi test).
    - Với các thành phố có quá ít mẫu, gộp thành nhóm "Other" để tránh
      one-hot sinh ra cột gần như toàn 0 (rất dễ overfit).
    """
    df = df.copy()

    if "statezip" in df.columns:
        df = df.drop(columns=["statezip"])

    kept_cities = get_kept_cities(df, min_count)
    df["city"] = df["city"].where(df["city"].isin(kept_cities), "Other")

    df = pd.get_dummies(df, columns=["city"], drop_first=True)
    return df


def save_schema(feature_columns: list, kept_cities: list, min_count: int) -> None:
    """Lưu lại đúng thứ tự + tên cột sau encode, cùng danh sách thành
    phố đã giữ lại -- để `predict.py` dựng lại y hệt input cho 1 căn
    nhà mới (bằng cách reindex về đúng bộ cột này), tránh lệch cột giữa
    lúc train và lúc dự đoán thực tế."""
    schema = {
        "feature_columns": list(feature_columns),
        "kept_cities": kept_cities,
        "min_city_count": min_count,
        "raw_required_fields": RAW_REQUIRED_FIELDS,
    }
    SCHEMA_PATH.parent.mkdir(parents=True, exist_ok=True)
    SCHEMA_PATH.write_text(json.dumps(schema, indent=2, ensure_ascii=False))
    print(f"Đã lưu schema vào {SCHEMA_PATH}")


def run_pipeline(save: bool = True) -> pd.DataFrame:
    df = load_raw_data()
    df = clean_data(df)
    df = engineer_features(df)

    kept_cities = get_kept_cities(df, MIN_CITY_COUNT)
    df = encode_categorical(df, MIN_CITY_COUNT)

    if save:
        PROCESSED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(PROCESSED_DATA_PATH, index=False)
        print(f"Đã lưu dữ liệu đã xử lý vào {PROCESSED_DATA_PATH}")

        feature_columns = [c for c in df.columns if c != "price"]
        save_schema(feature_columns, kept_cities, MIN_CITY_COUNT)

    return df


if __name__ == "__main__":
    df = run_pipeline()
    print(f"Shape sau xử lý: {df.shape}")
    print(df.head())
