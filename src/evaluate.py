"""
evaluate.py
Đánh giá chi tiết model đã train: residual plot, feature importance.
Chạy trực tiếp: python src/evaluate.py
"""

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split

from data_loader import load_processed_data

MODEL_PATH = Path("models/best_model.pkl")
FIG_DIR = Path("outputs/figures")


def plot_predicted_vs_actual(y_true, y_pred):
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.3, s=10)
    lims = [0, max(y_true.max(), y_pred.max())]
    plt.plot(lims, lims, "r--")
    plt.xlabel("Giá thực tế")
    plt.ylabel("Giá dự đoán")
    plt.title("Predicted vs Actual")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "predicted_vs_actual.png", dpi=110)


def plot_feature_importance(model, feature_names, top_n=15):
    if not hasattr(model, "feature_importances_"):
        print("Model này không có feature_importances_ (VD: LinearRegression).")
        return
    importances = pd.Series(model.feature_importances_, index=feature_names)
    importances = importances.sort_values(ascending=False).head(top_n)

    plt.figure(figsize=(8, 6))
    importances.sort_values().plot(kind="barh")
    plt.title(f"Top {top_n} Feature Importance")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "feature_importance.png", dpi=110)


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    df = load_processed_data()

    X = df.drop(columns=["price"])
    y = np.log1p(df["price"])

    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = joblib.load(MODEL_PATH)
    preds_log = model.predict(X_test)

    y_true = np.expm1(y_test)
    preds = np.expm1(preds_log)

    plot_predicted_vs_actual(y_true, preds)
    plot_feature_importance(model, X.columns)

    print(f"Đã lưu biểu đồ đánh giá vào {FIG_DIR}/")


if __name__ == "__main__":
    main()
