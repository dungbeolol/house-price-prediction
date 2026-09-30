import json

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.model_selection import train_test_split

pytest.importorskip("xgboost")
pytest.importorskip("catboost")

import train  # noqa: E402

SINGLE_MODELS = {"LinearRegression", "RidgeCV", "DecisionTree", "RandomForest",
                 "ExtraTrees", "GradientBoosting", "XGBoost", "CatBoost"}


@pytest.fixture
def xy(processed):
    df, _ = processed
    X = df.drop(columns=["price"])
    y = np.log1p(df["price"])
    return X, y


def test_get_models_returns_singles_plus_stacking(cfg):
    cfg["stacking"]["enabled"] = True
    models = train.get_models(cfg)
    assert set(models) == SINGLE_MODELS | {"Stacking"}
    assert list(models)[-1] == "Stacking"
    assert isinstance(models["Stacking"], StackingRegressor)


def test_stacking_can_be_disabled(cfg):
    cfg["stacking"]["enabled"] = False
    assert set(train.get_models(cfg)) == SINGLE_MODELS


def test_models_use_hyperparameters_from_config(cfg):
    cfg["models"]["random_forest"]["n_estimators"] = 7
    cfg["models"]["decision_tree"]["max_depth"] = 3
    cfg["seed"] = 123
    models = train.build_base_models(cfg)

    assert models["RandomForest"].n_estimators == 7
    assert models["DecisionTree"].max_depth == 3
    assert models["DecisionTree"].random_state == 123


def test_stacking_uses_base_models_from_config(cfg):
    cfg["stacking"]["base_models"] = ["RidgeCV", "ExtraTrees"]
    stack = train.build_stacking(train.build_base_models(cfg), cfg)
    assert [name for name, _ in stack.estimators] == ["RidgeCV", "ExtraTrees"]
    assert isinstance(stack.final_estimator, RidgeCV)


def test_stacking_clones_base_models(cfg):
    """Stacking không được dùng chung object với model đơn (fit 1 bên
    không được làm bẩn bên kia)."""
    base = train.build_base_models(cfg)
    stack = train.build_stacking(base, cfg)
    for name, est in stack.estimators:
        assert est is not base[name]


def test_stacking_unknown_base_model_raises(cfg):
    cfg["stacking"]["base_models"] = ["RidgeCV", "MoHinhKhongTonTai"]
    with pytest.raises(ValueError, match="MoHinhKhongTonTai"):
        train.build_stacking(train.build_base_models(cfg), cfg)


@pytest.mark.parametrize("names", [["CatBoost"], ["XGBoost", "XGBoost"], []])
def test_stacking_needs_two_distinct_base_models(cfg, names):
    cfg["stacking"]["base_models"] = names
    with pytest.raises(ValueError, match=">= 2"):
        train.build_stacking(train.build_base_models(cfg), cfg)


def test_stacking_fit_predict(small_cfg, xy):
    X, y = xy
    stack = train.get_models(small_cfg)["Stacking"]
    stack.fit(X, y)
    preds = stack.predict(X)

    assert preds.shape == (len(X),)
    assert np.isfinite(preds).all()
    # model cấp 2 có đúng 1 trọng số cho mỗi base model
    n_base = len(small_cfg["stacking"]["base_models"])
    assert len(stack.final_estimator_.coef_) == n_base


def test_stacking_beats_a_trivial_baseline(small_cfg, xy):
    """Sanity check: stacking phải tốt hơn hẳn việc luôn đoán giá trung bình."""
    X, y = xy
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=0)
    stack = train.get_models(small_cfg)["Stacking"].fit(X_tr, y_tr)

    rmse_stack = np.sqrt(np.mean((y_te - stack.predict(X_te)) ** 2))
    rmse_mean = np.sqrt(np.mean((y_te - y_tr.mean()) ** 2))
    assert rmse_stack < 0.7 * rmse_mean


def test_stacking_is_reproducible_with_same_seed(small_cfg, xy):
    X, y = xy
    small_cfg["stacking"]["base_models"] = ["RidgeCV", "ExtraTrees"]  # nhanh, ổn định
    p1 = train.get_models(small_cfg)["Stacking"].fit(X, y).predict(X)
    p2 = train.get_models(small_cfg)["Stacking"].fit(X, y).predict(X)
    np.testing.assert_allclose(p1, p2)


def test_cross_validate_models_returns_sorted_table(monkeypatch, xy):
    X, y = xy
    monkeypatch.setattr(train, "get_models", lambda cfg=None: {
        "Linear": LinearRegression(), "Ridge": RidgeCV(),
    })
    out = train.cross_validate_models(X.astype(float), y, n_splits=3)

    assert list(out.columns) == ["model", "CV_RMSLE_mean", "CV_RMSLE_std"]
    assert set(out["model"]) == {"Linear", "Ridge"}
    assert out["CV_RMSLE_mean"].is_monotonic_increasing


def test_evaluate_on_holdout_saves_best_model_and_meta(monkeypatch, tmp_path, xy):
    X, y = xy
    X = X.astype(float)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=0)

    monkeypatch.setattr(train, "MODEL_PATH", tmp_path / "m" / "best.pkl")
    monkeypatch.setattr(train, "META_PATH", tmp_path / "m" / "meta.json")
    monkeypatch.setattr(train, "get_models", lambda cfg=None: {
        "Linear": LinearRegression(), "Ridge": RidgeCV(),
    })

    results = train.evaluate_on_holdout(X_tr, X_te, y_tr, y_te)

    assert {"model", "RMSLE", "R2_log", "MAE_usd", "MdAPE_%"} <= set(results.columns)
    assert results["RMSLE"].is_monotonic_increasing

    saved = joblib.load(tmp_path / "m" / "best.pkl")
    assert saved.predict(X_te).shape == (len(X_te),)

    meta = json.loads((tmp_path / "m" / "meta.json").read_text())
    assert meta["model"] == results.iloc[0]["model"]   # model tốt nhất = hàng đầu
    assert meta["metrics"]["holdout_RMSLE"] == pytest.approx(results.iloc[0]["RMSLE"])
