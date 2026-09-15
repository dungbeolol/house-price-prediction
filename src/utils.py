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
