import json

import pytest
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import utils


def test_get_final_estimator_unwraps_pipeline():
    inner = LinearRegression()
    pipe = Pipeline([("scaler", StandardScaler()), ("model", inner)])
    assert utils.get_final_estimator(pipe) is inner


def test_get_final_estimator_plain_model():
    model = LinearRegression()
    assert utils.get_final_estimator(model) is model


def test_library_versions_has_tracked_packages():
    versions = utils.library_versions()
    assert set(versions) == set(utils.TRACKED_PACKAGES)
    assert versions["scikit-learn"]  # scikit-learn chắc chắn đã cài


def test_save_model_meta_roundtrip(tmp_path):
    path = tmp_path / "sub" / "meta.json"
    utils.save_model_meta(path, "CatBoost", 42, {"RMSLE": 0.29})

    meta = json.loads(path.read_text())
    assert meta["model"] == "CatBoost"
    assert meta["seed"] == 42
    assert meta["metrics"] == {"RMSLE": 0.29}
    assert "scikit-learn" in meta["library_versions"]


def test_check_model_compat_detects_mismatch():
    meta = {"library_versions": {"scikit-learn": "0.0.1"}}
    problems = utils.check_model_compat(meta)
    assert len(problems) == 1 and "scikit-learn" in problems[0]


def test_check_model_compat_ok_when_same_or_missing():
    assert utils.check_model_compat({"library_versions": utils.library_versions()}) == []
    assert utils.check_model_compat({}) == []
    assert utils.check_model_compat(None) == []
    # thư viện không cài (None) thì không tính là lệch
    assert utils.check_model_compat({"library_versions": {"xgboost": None}}) == []


def test_warn_if_incompatible_emits_runtime_warning():
    with pytest.warns(RuntimeWarning):
        utils.warn_if_incompatible({"library_versions": {"numpy": "0.0.1"}})
