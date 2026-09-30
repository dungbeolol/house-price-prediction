import copy

import pytest
import yaml

from config import CFG, PROJECT_ROOT, load_config, resolve_path, validate_config


def test_default_config_is_valid():
    validate_config(CFG)
    assert CFG["stacking"]["enabled"] in (True, False)


def test_paths_resolve_under_project_root():
    for key, value in CFG["paths"].items():
        resolved = resolve_path(value)
        assert resolved.is_absolute(), key
        assert PROJECT_ROOT in resolved.parents, key


def test_resolve_path_keeps_absolute_paths(tmp_path):
    assert resolve_path(tmp_path) == tmp_path


def test_missing_section_raises(cfg):
    del cfg["stacking"]
    with pytest.raises(ValueError, match="stacking"):
        validate_config(cfg)


def test_missing_key_in_section_raises(cfg):
    del cfg["paths"]["model"]
    with pytest.raises(ValueError, match="model"):
        validate_config(cfg)


def test_missing_seed_raises(cfg):
    del cfg["seed"]
    with pytest.raises(ValueError, match="seed"):
        validate_config(cfg)


@pytest.mark.parametrize("bad", [0, 1, -0.2, 1.5])
def test_invalid_test_size_raises(cfg, bad):
    cfg["data"]["test_size"] = bad
    with pytest.raises(ValueError, match="test_size"):
        validate_config(cfg)


def test_invalid_cv_splits_raises(cfg):
    cfg["cv"]["n_splits"] = 1
    with pytest.raises(ValueError, match="n_splits"):
        validate_config(cfg)


def test_empty_yaml_raises():
    with pytest.raises(ValueError):
        validate_config(None)


def test_load_config_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "khong_ton_tai.yaml")


def test_load_config_from_env_var(monkeypatch, tmp_path):
    custom = copy.deepcopy(CFG)
    custom["seed"] = 7
    path = tmp_path / "custom.yaml"
    path.write_text(yaml.safe_dump(custom))

    monkeypatch.setenv("HPP_CONFIG", str(path))
    assert load_config()["seed"] == 7
