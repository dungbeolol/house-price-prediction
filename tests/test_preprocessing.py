import json

import numpy as np
import pandas as pd
import pytest

import preprocessing as prep
from data_loader import RAW_DATA_PATH


def test_clean_data_drops_zero_price_leakage_and_street(raw_df):
    out = prep.clean_data(raw_df)

    assert (out["price"] > 0).all()
    assert len(out) == len(raw_df) - 2          # 2 dòng price = 0 bị bỏ
    assert "price_per_sqft" not in out.columns  # leakage
    assert "street" not in out.columns
    assert out.index.tolist() == list(range(len(out)))  # index được reset


def test_clean_data_does_not_mutate_input(raw_df):
    before = raw_df.copy()
    prep.clean_data(raw_df)
    pd.testing.assert_frame_equal(raw_df, before)


def test_engineer_features_values(cleaned_raw):
    out = prep.engineer_features(cleaned_raw)

    assert "date" not in out.columns
    assert {"year_sold", "month_sold", "house_age", "was_renovated",
            "years_since_renovation", "total_rooms"} <= set(out.columns)

    assert (out["house_age"] == out["year_sold"] - out["yr_built"]).all()
    assert (out["total_rooms"] == out["bedrooms"] + out["bathrooms"]).all()

    renovated = out["yr_renovated"] > 0
    assert (out["was_renovated"] == renovated.astype(int)).all()
    # đã sửa: tính từ năm sửa; chưa sửa: bằng tuổi nhà
    assert (out.loc[renovated, "years_since_renovation"]
            == out.loc[renovated, "year_sold"] - out.loc[renovated, "yr_renovated"]).all()
    assert (out.loc[~renovated, "years_since_renovation"]
            == out.loc[~renovated, "house_age"]).all()


def test_get_kept_cities_respects_threshold(cleaned_raw):
    df = prep.engineer_features(cleaned_raw)
    assert prep.get_kept_cities(df, min_count=30) == ["Bellevue", "Kent", "Seattle"]
    assert prep.get_kept_cities(df, min_count=5) == ["Bellevue", "Kent", "Seattle", "Tiny1"]
    assert prep.get_kept_cities(df, min_count=10_000) == []


def test_encode_categorical_merges_rare_cities_and_drops_statezip(cleaned_raw):
    df = prep.engineer_features(cleaned_raw)
    out = prep.encode_categorical(df, min_count=30)

    assert "statezip" not in out.columns
    assert "city" not in out.columns
    # thành phố hiếm không có cột riêng, đã gộp vào Other
    assert "city_Tiny1" not in out.columns and "city_Tiny2" not in out.columns
    assert "city_Other" in out.columns
    # drop_first: thành phố đầu bảng chữ cái (Bellevue) là baseline, không có cột
    assert "city_Bellevue" not in out.columns
    assert {"city_Kent", "city_Seattle"} <= set(out.columns)


def test_encode_categorical_no_missing_values(cleaned_raw):
    out = prep.encode_categorical(prep.engineer_features(cleaned_raw), min_count=30)
    assert not out.isna().any().any()


def test_run_pipeline_without_saving(monkeypatch, raw_df):
    monkeypatch.setattr(prep, "load_raw_data", lambda: raw_df)
    out = prep.run_pipeline(save=False)

    assert "price_per_sqft" not in out.columns
    assert (out["price"] > 0).all()
    assert not out.isna().any().any()


def test_run_pipeline_saves_processed_data_and_schema(monkeypatch, raw_df, tmp_path):
    processed_path = tmp_path / "out" / "cleaned.csv"
    schema_path = tmp_path / "models" / "schema.json"
    monkeypatch.setattr(prep, "load_raw_data", lambda: raw_df)
    monkeypatch.setattr(prep, "PROCESSED_DATA_PATH", processed_path)
    monkeypatch.setattr(prep, "SCHEMA_PATH", schema_path)

    out = prep.run_pipeline(save=True)

    assert processed_path.exists()
    assert len(pd.read_csv(processed_path)) == len(out)

    schema = json.loads(schema_path.read_text())
    assert "price" not in schema["feature_columns"]
    assert schema["kept_cities"] == ["Bellevue", "Kent", "Seattle"]
    assert schema["min_city_count"] == prep.MIN_CITY_COUNT
    assert schema["raw_required_fields"] == prep.RAW_REQUIRED_FIELDS


@pytest.mark.skipif(not RAW_DATA_PATH.exists(), reason="Không có data/raw (bị .gitignore)")
def test_pipeline_on_real_data_has_no_leakage():
    out = prep.run_pipeline(save=False)

    assert len(out) > 1000
    assert "price_per_sqft" not in out.columns
    assert "statezip" not in out.columns
    assert (out["price"] > 0).all()
    assert np.isfinite(out.select_dtypes("number")).all().all()
