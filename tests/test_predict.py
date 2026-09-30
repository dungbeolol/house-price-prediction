import json
import warnings

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

import predict
import preprocessing as prep


def test_build_features_matches_schema_columns_and_order(house, schema):
    X = predict.build_features(house, schema)

    assert X.shape == (1, len(schema["feature_columns"]))
    assert list(X.columns) == schema["feature_columns"]
    assert not X.isna().any().any()


def test_unknown_city_is_mapped_to_other(house, schema):
    house_unknown = {**house, "city": "ThanhPhoLaXaKhongCoTrongData"}
    house_rare = {**house, "city": "Tiny1"}   # có trong data nhưng < ngưỡng

    X_unknown = predict.build_features(house_unknown, schema)
    X_rare = predict.build_features(house_rare, schema)

    assert X_unknown["city_Other"].iloc[0] == 1
    pd.testing.assert_frame_equal(X_unknown, X_rare)


def test_baseline_city_has_all_city_columns_zero(house, schema):
    """Bellevue là baseline (bị drop_first lúc train) -> mọi cột city_* = 0."""
    X = predict.build_features({**house, "city": "Bellevue"}, schema)
    city_cols = [c for c in X.columns if c.startswith("city_")]
    assert city_cols and (X[city_cols].iloc[0] == 0).all()


def test_missing_required_field_raises(house, schema):
    incomplete = {k: v for k, v in house.items() if k != "sqft_living"}
    with pytest.raises(ValueError, match="sqft_living"):
        predict.build_features(incomplete, schema)


def test_extra_fields_are_ignored(house, schema):
    X = predict.build_features({**house, "ghi_chu": "khong dung"}, schema)
    assert list(X.columns) == schema["feature_columns"]


def test_train_serve_consistency(cleaned_raw, processed, schema):
    """Test quan trọng nhất: cùng 1 căn nhà đi qua pipeline batch (lúc train)
    và qua build_features (lúc predict) phải cho ĐÚNG cùng vector feature.
    Nếu ai đó sửa preprocessing mà quên predict (hoặc ngược lại), test này đỏ."""
    batch_df, _ = processed
    X_batch = batch_df.drop(columns=["price"])[schema["feature_columns"]]

    rng = np.random.default_rng(1)
    for i in rng.choice(len(cleaned_raw), size=25, replace=False):
        raw = cleaned_raw.loc[i, prep.RAW_REQUIRED_FIELDS].to_dict()
        X_single = predict.build_features(raw, schema)

        np.testing.assert_allclose(
            X_single.iloc[0].astype(float).to_numpy(),
            X_batch.iloc[i].astype(float).to_numpy(),
            err_msg=f"Lệch feature ở dòng {i} (city={raw['city']})",
        )


def test_predict_price_returns_positive_finite_float(house, processed, schema):
    df, _ = processed
    X = df.drop(columns=["price"])
    model = LinearRegression().fit(X.astype(float), np.log1p(df["price"]))

    price = predict.predict_price(house, model=model, schema=schema)

    assert isinstance(price, float)
    assert np.isfinite(price) and price > 0


def _dump_artifacts(tmp_path, processed, schema, meta=None):
    df, _ = processed
    X = df.drop(columns=["price"]).astype(float)
    model = LinearRegression().fit(X, np.log1p(df["price"]))
    joblib.dump(model, tmp_path / "model.pkl")
    (tmp_path / "schema.json").write_text(json.dumps(schema))
    if meta is not None:
        (tmp_path / "meta.json").write_text(json.dumps(meta))


def _point_predict_at(monkeypatch, tmp_path):
    monkeypatch.setattr(predict, "MODEL_PATH", tmp_path / "model.pkl")
    monkeypatch.setattr(predict, "SCHEMA_PATH", tmp_path / "schema.json")
    monkeypatch.setattr(predict, "META_PATH", tmp_path / "meta.json")


def test_load_artifacts_missing_model_raises(monkeypatch, tmp_path):
    _point_predict_at(monkeypatch, tmp_path)
    with pytest.raises(FileNotFoundError, match="train.py"):
        predict.load_artifacts()


def test_load_artifacts_missing_schema_raises(monkeypatch, tmp_path, processed, schema):
    _dump_artifacts(tmp_path, processed, schema)
    (tmp_path / "schema.json").unlink()
    _point_predict_at(monkeypatch, tmp_path)
    with pytest.raises(FileNotFoundError, match="preprocessing.py"):
        predict.load_artifacts()


def test_load_artifacts_ok_without_meta_file(monkeypatch, tmp_path, processed, schema):
    """Model cũ (chưa có model_meta.json) vẫn phải load được, không cảnh báo."""
    _dump_artifacts(tmp_path, processed, schema, meta=None)
    _point_predict_at(monkeypatch, tmp_path)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        model, loaded_schema = predict.load_artifacts()
    assert loaded_schema["feature_columns"] == schema["feature_columns"]


def test_load_artifacts_warns_on_version_mismatch(monkeypatch, tmp_path, processed, schema):
    meta = {"library_versions": {"scikit-learn": "0.0.1"}}
    _dump_artifacts(tmp_path, processed, schema, meta=meta)
    _point_predict_at(monkeypatch, tmp_path)

    with pytest.warns(RuntimeWarning, match="scikit-learn"):
        predict.load_artifacts()


def test_predict_many_runs_each_house(monkeypatch, tmp_path, processed, schema, house):
    _dump_artifacts(tmp_path, processed, schema)
    _point_predict_at(monkeypatch, tmp_path)

    prices = predict.predict_many([house, {**house, "city": "Kent"}])
    assert len(prices) == 2 and all(p > 0 for p in prices)
