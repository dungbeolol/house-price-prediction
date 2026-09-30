"""
utils.py
Các hàm dùng chung cho nhiều bước trong pipeline.
"""

import json
import warnings
from datetime import datetime, timezone
from importlib import metadata

import matplotlib.pyplot as plt

# Thư viện ảnh hưởng trực tiếp tới việc đọc lại file model.pkl
TRACKED_PACKAGES = ["scikit-learn", "xgboost", "catboost", "pandas", "numpy"]


def save_fig(fig, path, dpi=110):
    """Lưu figure với thư mục tự tạo nếu chưa có."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    print(f"Đã lưu biểu đồ: {path}")


def print_section(title: str):
    """In tiêu đề section cho dễ đọc log khi chạy script."""
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def get_final_estimator(model):
    """Trả về estimator cuối cùng, dù model là Pipeline hay estimator thường.

    Cần thiết vì train.py có thể lưu Pipeline(scaler + model) cho
    LinearRegression/RidgeCV, nhưng evaluate.py muốn lấy trực tiếp
    coef_/feature_importances_ của model bên trong.
    """
    if hasattr(model, "named_steps"):
        return model.named_steps["model"]
    return model


def library_versions() -> dict:
    """Phiên bản các thư viện quan trọng đang cài (None nếu chưa cài)."""
    versions = {}
    for pkg in TRACKED_PACKAGES:
        try:
            versions[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            versions[pkg] = None
    return versions


def save_model_meta(path, model_name: str, seed: int, metrics: dict = None) -> None:
    """Ghi lại model nào được lưu, khi nào, với phiên bản thư viện nào.

    Vì sao cần: file .pkl chỉ đọc lại an toàn với đúng phiên bản
    scikit-learn/xgboost/catboost đã dùng lúc lưu. Ghi lại phiên bản để
    predict.py cảnh báo khi môi trường hiện tại lệch.
    """
    meta = {
        "model": model_name,
        "saved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seed": seed,
        "metrics": metrics or {},
        "library_versions": library_versions(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(meta, indent=2, ensure_ascii=False))


def check_model_compat(meta: dict) -> list:
    """So phiên bản thư viện lúc lưu model với môi trường hiện tại.
    Trả về danh sách thông báo lệch (rỗng nếu khớp hoặc không có meta)."""
    saved = (meta or {}).get("library_versions", {})
    current = library_versions()
    problems = []
    for pkg, saved_ver in saved.items():
        cur_ver = current.get(pkg)
        if saved_ver and cur_ver and saved_ver != cur_ver:
            problems.append(f"{pkg}: model lưu với {saved_ver}, đang chạy {cur_ver}")
    return problems


def warn_if_incompatible(meta: dict) -> None:
    problems = check_model_compat(meta)
    if problems:
        warnings.warn(
            "Phiên bản thư viện lệch so với lúc lưu model (" + "; ".join(problems)
            + "). Kết quả có thể sai hoặc lỗi khi load -- nên train lại model.",
            RuntimeWarning, stacklevel=2,
        )
