"""
utils.py
Các hàm dùng chung cho nhiều bước trong pipeline.
"""

import matplotlib.pyplot as plt


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
