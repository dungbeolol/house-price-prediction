"""
data_loader.py
Hàm đọc dữ liệu thô và dữ liệu đã xử lý.
"""

import pandas as pd
from pathlib import Path

RAW_DATA_PATH = Path("data/raw/modified_data.csv")
PROCESSED_DATA_PATH = Path("data/processed/cleaned_data.csv")


def load_raw_data(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Đọc dữ liệu gốc, chưa qua xử lý."""
    df = pd.read_csv(path)
    return df


def load_processed_data(path: Path = PROCESSED_DATA_PATH) -> pd.DataFrame:
    """Đọc dữ liệu đã làm sạch (output của preprocessing.py)."""
    if not path.exists():
        raise FileNotFoundError(
            f"Chưa có {path}. Hãy chạy src/preprocessing.py trước."
        )
    return pd.read_csv(path)


if __name__ == "__main__":
    df = load_raw_data()
    print(f"Đã load {df.shape[0]} dòng, {df.shape[1]} cột.")
    print(df.head())
