"""
config.py
Đọc config.yaml một lần và cung cấp cho toàn bộ project.

    from config import CFG, resolve_path
    resolve_path(CFG["paths"]["model"])   # -> đường dẫn tuyệt đối

Đường dẫn trong config luôn được hiểu tương đối so với thư mục gốc
project (PROJECT_ROOT), không phụ thuộc thư mục hiện tại khi chạy script.
"""

import os
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"

# Các mục bắt buộc phải có -> báo lỗi sớm, rõ ràng nếu config thiếu/sai
REQUIRED_SECTIONS = {
    "paths": ["raw_data", "processed_data", "model", "schema",
              "model_meta", "figures_dir", "reports_dir"],
    "data": ["target", "test_size"],
    "preprocessing": ["min_city_count"],
    "cv": ["n_splits"],
    "models": ["ridge", "decision_tree", "random_forest", "extra_trees",
               "gradient_boosting", "xgboost", "catboost"],
    "stacking": ["enabled", "base_models", "final_alphas", "cv_splits",
                 "passthrough"],
    "tuning": ["n_iter", "search_cv_splits"],
}


def validate_config(cfg: dict) -> None:
    """Kiểm tra cấu trúc config; raise ValueError kèm chỗ sai cụ thể."""
    if not isinstance(cfg, dict):
        raise ValueError("config.yaml rỗng hoặc không đúng định dạng YAML.")

    for key in ("seed", "n_jobs"):
        if key not in cfg:
            raise ValueError(f"config thiếu khóa cấp cao nhất: '{key}'")

    for section, keys in REQUIRED_SECTIONS.items():
        if section not in cfg or not isinstance(cfg[section], dict):
            raise ValueError(f"config thiếu mục: '{section}'")
        missing = [k for k in keys if k not in cfg[section]]
        if missing:
            raise ValueError(f"config['{section}'] thiếu: {missing}")

    if not 0 < cfg["data"]["test_size"] < 1:
        raise ValueError("data.test_size phải nằm trong khoảng (0, 1).")
    for name, n in (("cv.n_splits", cfg["cv"]["n_splits"]),
                    ("stacking.cv_splits", cfg["stacking"]["cv_splits"]),
                    ("tuning.search_cv_splits", cfg["tuning"]["search_cv_splits"])):
        if not isinstance(n, int) or n < 2:
            raise ValueError(f"{name} phải là số nguyên >= 2.")


def load_config(path=None) -> dict:
    """Đọc + kiểm tra config. Ưu tiên: tham số path > biến môi trường
    HPP_CONFIG > config.yaml ở thư mục gốc."""
    path = Path(path or os.environ.get("HPP_CONFIG") or DEFAULT_CONFIG_PATH)
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy file config: {path}")
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    validate_config(cfg)
    return cfg


def resolve_path(p) -> Path:
    """Đường dẫn trong config -> đường dẫn tuyệt đối (gốc = PROJECT_ROOT)."""
    p = Path(p)
    return p if p.is_absolute() else PROJECT_ROOT / p


CFG = load_config()
