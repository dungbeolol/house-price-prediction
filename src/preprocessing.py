"""
preprocessing.py
Làm sạch dữ liệu + tạo feature mới cho bài toán dự đoán giá nhà.
Chạy trực tiếp: python src/preprocessing.py
"""

import pandas as pd
import numpy as np
from pathlib import Path

from data_loader import load_raw_data, PROCESSED_DATA_PATH


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


def encode_categorical(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode các cột dạng chuỗi (city, statezip)."""
    df = df.copy()
    df = pd.get_dummies(df, columns=["city", "statezip"], drop_first=True)
    return df


def run_pipeline(save: bool = True) -> pd.DataFrame:
    df = load_raw_data()
    df = clean_data(df)
    df = engineer_features(df)
    df = encode_categorical(df)

    if save:
        PROCESSED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(PROCESSED_DATA_PATH, index=False)
        print(f"Đã lưu dữ liệu đã xử lý vào {PROCESSED_DATA_PATH}")

    return df


if __name__ == "__main__":
    df = run_pipeline()
    print(f"Shape sau xử lý: {df.shape}")
    print(df.head())
