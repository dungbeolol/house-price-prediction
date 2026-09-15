"""
train.py
Train và so sánh nhiều model cho bài toán dự đoán giá nhà.
Chạy trực tiếp: python src/train.py
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from data_loader import load_processed_data

MODEL_DIR = Path("models")
TARGET_COL = "price"


def get_models() -> dict:
    return {
        "LinearRegression": LinearRegression(),
        "Ridge": Ridge(alpha=1.0),
        "RandomForest": RandomForestRegressor(
            n_estimators=300, max_depth=12, random_state=42, n_jobs=-1
        ),
    }


def train_and_compare():
    df = load_processed_data()

    X = df.drop(columns=[TARGET_COL])
    y = np.log1p(df[TARGET_COL])  # log-transform target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    results = []
    best_model = None
    best_rmse = np.inf

    for name, model in get_models().items():
        model.fit(X_train, y_train)
        preds_log = model.predict(X_test)

        # Đưa về thang giá trị gốc (USD) để dễ hiểu
        preds = np.expm1(preds_log)
        y_true = np.expm1(y_test)

        rmse = np.sqrt(mean_squared_error(y_true, preds))
        mae = mean_absolute_error(y_true, preds)
        r2 = r2_score(y_true, preds)

        results.append({"model": name, "RMSE": rmse, "MAE": mae, "R2": r2})
        print(f"{name:20s} RMSE={rmse:,.0f}  MAE={mae:,.0f}  R2={r2:.3f}")

        if rmse < best_rmse:
            best_rmse = rmse
            best_model = model

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(best_model, MODEL_DIR / "best_model.pkl")
    print(f"\nĐã lưu model tốt nhất vào {MODEL_DIR / 'best_model.pkl'}")

    results_df = pd.DataFrame(results).sort_values("RMSE")
    results_df.to_csv("outputs/reports/model_comparison.csv", index=False)
    return results_df


if __name__ == "__main__":
    train_and_compare()
