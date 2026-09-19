"""
tune.py
Phần 6: Hyperparameter tuning cho XGBoost và RandomForest bằng
RandomizedSearchCV, so sánh với baseline (tham số mặc định ở Phần 4).
Chạy trực tiếp: python src/tune.py
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from scipy.stats import randint, uniform

from sklearn.model_selection import KFold, RandomizedSearchCV, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from data_loader import load_processed_data

MODEL_DIR = Path("models")
REPORT_DIR = Path("outputs/reports")

# RandomizedSearchCV thay vì GridSearchCV: với nhiều tham số, thử TẤT CẢ
# tổ hợp (Grid) sẽ rất chậm; Random chỉ thử ngẫu nhiên N tổ hợp trong
# không gian tham số, thường tìm được vùng tốt gần tương đương mà nhanh
# hơn nhiều -> phù hợp cho dataset/máy cá nhân.

XGB_PARAM_DIST = {
    "n_estimators": randint(150, 500),
    "max_depth": randint(3, 7),
    "learning_rate": uniform(0.02, 0.18),       # 0.02 -> 0.20
    "subsample": uniform(0.7, 0.3),             # 0.7 -> 1.0
    "colsample_bytree": uniform(0.7, 0.3),      # 0.7 -> 1.0
    "min_child_weight": randint(1, 6),
}

RF_PARAM_DIST = {
    "n_estimators": randint(150, 400),
    "max_depth": randint(5, 20),
    "min_samples_leaf": randint(1, 8),
    "max_features": uniform(0.3, 0.7),          # tỉ lệ số feature mỗi split
}


def tune_model(estimator, param_dist, X, y, n_iter=15, name=""):
    kfold = KFold(n_splits=3, shuffle=True, random_state=42)
    search = RandomizedSearchCV(
        estimator=estimator,
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring="neg_mean_squared_error",  # tương ứng RMSLE vì y đã log1p
        cv=kfold,
        random_state=42,
        n_jobs=-1,
        verbose=0,
    )
    search.fit(X, y)
    best_rmsle = np.sqrt(-search.best_score_)
    print(f"\n[{name}] Best RMSLE (CV) = {best_rmsle:.4f}")
    print(f"[{name}] Best params: {search.best_params_}")
    return search.best_estimator_, best_rmsle, search.best_params_


def baseline_rmsle(estimator, X, y) -> float:
    """RMSLE của model với tham số mặc định (Phần 4), để so sánh."""
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    neg_mse = cross_val_score(
        estimator, X, y, cv=kfold, scoring="neg_mean_squared_error", n_jobs=-1
    )
    return np.sqrt(-neg_mse).mean()


def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_processed_data()
    X = df.drop(columns=["price"])
    y = np.log1p(df["price"])

    # --- XGBoost: baseline (Phần 4) vs tuned ---
    xgb_baseline = XGBRegressor(
        n_estimators=400, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1,
    )
    xgb_base_rmsle = baseline_rmsle(xgb_baseline, X, y)
    print(f"[XGBoost] Baseline RMSLE (CV) = {xgb_base_rmsle:.4f}")

    xgb_tuned, xgb_tuned_rmsle, xgb_params = tune_model(
        XGBRegressor(random_state=42, n_jobs=-1), XGB_PARAM_DIST, X, y,
        n_iter=15, name="XGBoost"
    )

    # --- RandomForest: baseline vs tuned ---
    rf_baseline = RandomForestRegressor(
        n_estimators=300, max_depth=12, random_state=42, n_jobs=-1
    )
    rf_base_rmsle = baseline_rmsle(rf_baseline, X, y)
    print(f"\n[RandomForest] Baseline RMSLE (CV) = {rf_base_rmsle:.4f}")

    rf_tuned, rf_tuned_rmsle, rf_params = tune_model(
        RandomForestRegressor(random_state=42, n_jobs=-1), RF_PARAM_DIST, X, y,
        n_iter=15, name="RandomForest"
    )

    # So sánh công bằng bằng 5-fold cho cả model đã lưu ở Phần 4 và các
    # model vừa tune (tune_model() ở trên dùng 3-fold cho nhanh, nhưng
    # quyết định CUỐI CÙNG nên dùng cùng 1 chuẩn 5-fold cho mọi model)
    prev_model = joblib.load(MODEL_DIR / "best_model.pkl")
    prev_rmsle = baseline_rmsle(prev_model, X, y)
    xgb_tuned_rmsle_5fold = baseline_rmsle(xgb_tuned, X, y)
    rf_tuned_rmsle_5fold = baseline_rmsle(rf_tuned, X, y)

    final_compare = pd.DataFrame([
        {"model": "best_model.pkl (đã lưu)", "RMSLE_5fold": prev_rmsle},
        {"model": "XGBoost_tuned (mới)", "RMSLE_5fold": xgb_tuned_rmsle_5fold},
        {"model": "RandomForest_tuned (mới)", "RMSLE_5fold": rf_tuned_rmsle_5fold},
    ]).sort_values("RMSLE_5fold")
    final_compare.to_csv(REPORT_DIR / "tuning_comparison.csv", index=False)
    print("\n=== So sánh cuối (5-fold, công bằng) ===")
    print(final_compare.to_string(index=False))

    best_row = final_compare.iloc[0]
    print(f"\nModel tốt nhất: {best_row['model']}")

    if best_row["model"] == "XGBoost_tuned (mới)":
        xgb_tuned.fit(X, y)
        joblib.dump(xgb_tuned, MODEL_DIR / "best_model.pkl")
        print("Đã cập nhật models/best_model.pkl bằng XGBoost đã tune.")
    else:
        print("Model đã lưu vẫn tốt nhất/gần bằng -> giữ nguyên best_model.pkl.")


if __name__ == "__main__":
    main()
