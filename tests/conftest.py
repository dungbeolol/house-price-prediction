"""
conftest.py
Fixture dùng chung cho các test. Dữ liệu là DỮ LIỆU GIẢ (synthetic) sinh
ngay trong test, nên test chạy được mà không cần data/ và không phụ thuộc
file model đã train.
"""

import copy
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# src/ dùng import phẳng (from data_loader import ...) -> thêm vào sys.path
SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config import CFG  # noqa: E402
import preprocessing as prep  # noqa: E402

# Số nhà theo thành phố: 3 thành phố >= 30 mẫu (được giữ), 2 thành phố hiếm
# (< 30 mẫu, phải bị gộp vào "Other")
CITY_COUNTS = {"Seattle": 120, "Bellevue": 90, "Kent": 60, "Tiny1": 6, "Tiny2": 4}
CITY_FACTOR = {"Seattle": 1.0, "Bellevue": 1.6, "Kent": 0.7, "Tiny1": 0.9, "Tiny2": 1.1}


def make_raw_df(seed: int = 0) -> pd.DataFrame:
    """DataFrame thô có cùng cột với data/raw/modified_data.csv, kèm 2 dòng
    price = 0 (lỗi nhập liệu) để test bước làm sạch."""
    rng = np.random.default_rng(seed)
    cities = np.repeat(list(CITY_COUNTS), list(CITY_COUNTS.values()))
    n = len(cities)

    sqft_living = rng.integers(600, 4000, n)
    sqft_basement = np.where(rng.random(n) < 0.3, rng.integers(100, 800, n), 0)
    yr_built = rng.integers(1920, 2014, n)
    renovated = rng.random(n) < 0.25
    yr_renovated = np.where(renovated, yr_built + rng.integers(5, 30, n), 0)
    factor = np.array([CITY_FACTOR[c] for c in cities])
    price = 120 * sqft_living * factor * rng.lognormal(0, 0.15, n) + 20000

    df = pd.DataFrame({
        "date": pd.to_datetime("2014-05-01") + pd.to_timedelta(rng.integers(0, 60, n), unit="D"),
        "price": price.round(0),
        "bedrooms": rng.integers(1, 6, n).astype(float),
        "bathrooms": rng.choice([1.0, 1.5, 2.0, 2.5, 3.0], n),
        "sqft_living": sqft_living,
        "sqft_lot": rng.integers(1500, 12000, n),
        "floors": rng.choice([1.0, 1.5, 2.0], n),
        "waterfront": (rng.random(n) < 0.05).astype(int),
        "view": rng.integers(0, 5, n),
        "condition": rng.integers(1, 6, n),
        "sqft_above": sqft_living - sqft_basement,
        "sqft_basement": sqft_basement,
        "yr_built": yr_built,
        "yr_renovated": yr_renovated,
        "street": [f"{i} Example St" for i in range(n)],
        "city": cities,
        "statezip": [f"WA 98{100 + list(CITY_COUNTS).index(c)}" for c in cities],
    })
    df["price_per_sqft"] = (df["price"] / df["sqft_living"]).round(2)
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")

    bad = df.iloc[:2].copy()
    bad["price"] = 0.0
    return pd.concat([df, bad], ignore_index=True)


@pytest.fixture
def raw_df() -> pd.DataFrame:
    return make_raw_df()


@pytest.fixture
def cleaned_raw(raw_df) -> pd.DataFrame:
    """Dữ liệu thô sau clean_data() (còn cột date/city, chưa encode)."""
    return prep.clean_data(raw_df)


@pytest.fixture
def processed(cleaned_raw):
    """(df đã xử lý đầy đủ, danh sách thành phố được giữ) -- đúng thứ tự
    các bước như preprocessing.run_pipeline()."""
    df = prep.engineer_features(cleaned_raw)
    kept = prep.get_kept_cities(df, prep.MIN_CITY_COUNT)
    return prep.encode_categorical(df, prep.MIN_CITY_COUNT), kept


@pytest.fixture
def schema(processed) -> dict:
    df, kept = processed
    return {
        "feature_columns": [c for c in df.columns if c != "price"],
        "kept_cities": kept,
        "min_city_count": prep.MIN_CITY_COUNT,
        "raw_required_fields": prep.RAW_REQUIRED_FIELDS,
    }


@pytest.fixture
def house() -> dict:
    return {
        "date": "2024-06-15",
        "bedrooms": 3, "bathrooms": 2, "sqft_living": 1800, "sqft_lot": 5000,
        "floors": 1, "waterfront": 0, "view": 0, "condition": 3,
        "sqft_above": 1800, "sqft_basement": 0,
        "yr_built": 1995, "yr_renovated": 0, "city": "Seattle",
    }


@pytest.fixture
def cfg() -> dict:
    """Bản sao config để test sửa tham số mà không ảnh hưởng CFG toàn cục."""
    return copy.deepcopy(CFG)


@pytest.fixture
def small_cfg(cfg) -> dict:
    """Config thu nhỏ (ít cây, ít fold) để test model chạy trong vài giây."""
    for name in ("random_forest", "extra_trees", "gradient_boosting",
                 "xgboost", "catboost"):
        cfg["models"][name]["n_estimators"] = 10
    cfg["stacking"]["cv_splits"] = 2
    return cfg
