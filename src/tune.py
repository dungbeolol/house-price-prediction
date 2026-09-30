"""
tune.py
Phần 6: Hyperparameter tuning cho 3 model boosting/ensemble mạnh nhất ở
Phần 4 (XGBoost, RandomForest, CatBoost) bằng RandomizedSearchCV, so
sánh với baseline (tham số mặc định ở Phần 4).
Chạy trực tiếp: python src/tune.py

Ghi chú: chỉ tune 3 model tiềm năng nhất (không tune cả 8 model của
Phần 4) để tiết kiệm thời gian -- DecisionTree/LinearRegression/Ridge
đã thua xa nhóm boosting nên tuning thêm khó vượt qua; ExtraTrees và
GradientBoosting có thể tune thêm sau nếu muốn mở rộng.
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from scipy.stats import randint, uniform

from sklearn.model_selection import KFold, RandomizedSearchCV, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

from config import CFG, resolve_path
from data_loader import load_processed_data
from utils import save_model_meta

MODEL_PATH = resolve_path(CFG["paths"]["model"])
META_PATH = resolve_path(CFG["paths"]["model_meta"])
REPORT_DIR = resolve_path(CFG["paths"]["reports_dir"])
SEED = CFG["seed"]
TARGET_COL = CFG["data"]["target"]

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

CATBOOST_PARAM_DIST = {
    "n_estimators": randint(200, 600),
    "depth": randint(4, 8),
    "learning_rate": uniform(0.02, 0.18),       # 0.02 -> 0.20
    "l2_leaf_reg": uniform(1, 9),               # 1 -> 10, regularization
    "subsample": uniform(0.7, 0.3),             # 0.7 -> 1.0
}


def tune_model(estimator, param_dist, X, y, n_iter=None, name=""):
    n_iter = n_iter or CFG["tuning"]["n_iter"]
    kfold = KFold(n_splits=CFG["tuning"]["search_cv_splits"],
                  shuffle=True, random_state=SEED)
    search = RandomizedSearchCV(
        estimator=estimator,
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring="neg_mean_squared_error",  # tương ứng RMSLE vì y đã log1p
        cv=kfold,
        random_state=SEED,
        n_jobs=CFG["n_jobs"],
        verbose=0,
    )
    search.fit(X, y)
    best_rmsle = np.sqrt(-search.best_score_)
    print(f"\n[{name}] Best RMSLE (CV) = {best_rmsle:.4f}")
    print(f"[{name}] Best params: {search.best_params_}")
    return search.best_estimator_, best_rmsle, search.best_params_


def baseline_rmsle(estimator, X, y) -> float:
    """RMSLE của model với tham số mặc định (Phần 4), để so sánh."""
    kfold = KFold(n_splits=CFG["cv"]["n_splits"], shuffle=True, random_state=SEED)
    neg_mse = cross_val_score(
        estimator, X, y, cv=kfold, scoring="neg_mean_squared_error",
        n_jobs=CFG["n_jobs"],
    )
    return np.sqrt(-neg_mse).mean()


def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_processed_data()
    X = df.drop(columns=[TARGET_COL])
    y = np.log1p(df[TARGET_COL])

    # --- XGBoost: baseline (Phần 4) vs tuned ---
    xgb_baseline = XGBRegressor(
        random_state=SEED, n_jobs=CFG["n_jobs"], **CFG["models"]["xgboost"]
    )
    xgb_base_rmsle = baseline_rmsle(xgb_baseline, X, y)
    print(f"[XGBoost] Baseline RMSLE (CV) = {xgb_base_rmsle:.4f}")

    xgb_tuned, xgb_tuned_rmsle, xgb_params = tune_model(
        XGBRegressor(random_state=SEED, n_jobs=CFG["n_jobs"]), XGB_PARAM_DIST,
        X, y, name="XGBoost"
    )

    # --- RandomForest: baseline vs tuned ---
    rf_baseline = RandomForestRegressor(
        random_state=SEED, n_jobs=CFG["n_jobs"], **CFG["models"]["random_forest"]
    )
    rf_base_rmsle = baseline_rmsle(rf_baseline, X, y)
    print(f"\n[RandomForest] Baseline RMSLE (CV) = {rf_base_rmsle:.4f}")

    rf_tuned, rf_tuned_rmsle, rf_params = tune_model(
        RandomForestRegressor(random_state=SEED, n_jobs=CFG["n_jobs"]),
        RF_PARAM_DIST, X, y, name="RandomForest"
    )

    # --- CatBoost: baseline (Phần 4, model tốt nhất) vs tuned ---
    # bootstrap_type="Bernoulli" cần khai báo cố định để "subsample" có
    # hiệu lực (mặc định CatBoost dùng "MVS", không nhận subsample).
    cb_baseline = CatBoostRegressor(
        random_state=SEED, verbose=False, **CFG["models"]["catboost"]
    )
    cb_base_rmsle = baseline_rmsle(cb_baseline, X, y)
    print(f"\n[CatBoost] Baseline RMSLE (CV) = {cb_base_rmsle:.4f}")

    cb_tuned, cb_tuned_rmsle, cb_params = tune_model(
        CatBoostRegressor(random_state=SEED, verbose=False,
                           bootstrap_type="Bernoulli"),
        CATBOOST_PARAM_DIST, X, y, name="CatBoost"
    )

    # So sánh công bằng bằng 5-fold cho model đã lưu ở Phần 4 và cả 3
    # model vừa tune (tune_model() ở trên dùng 3-fold cho nhanh, nhưng
    # quyết định CUỐI CÙNG nên dùng cùng 1 chuẩn 5-fold cho mọi model)
    prev_model = joblib.load(MODEL_PATH)
    prev_rmsle = baseline_rmsle(prev_model, X, y)
    xgb_tuned_rmsle_5fold = baseline_rmsle(xgb_tuned, X, y)
    rf_tuned_rmsle_5fold = baseline_rmsle(rf_tuned, X, y)
    cb_tuned_rmsle_5fold = baseline_rmsle(cb_tuned, X, y)

    final_compare = pd.DataFrame([
        {"model": "best_model.pkl (đã lưu)", "RMSLE_5fold": prev_rmsle},
        {"model": "XGBoost_tuned (mới)", "RMSLE_5fold": xgb_tuned_rmsle_5fold},
        {"model": "RandomForest_tuned (mới)", "RMSLE_5fold": rf_tuned_rmsle_5fold},
        {"model": "CatBoost_tuned (mới)", "RMSLE_5fold": cb_tuned_rmsle_5fold},
    ]).sort_values("RMSLE_5fold")
    final_compare.to_csv(REPORT_DIR / "tuning_comparison.csv", index=False)
    print("\n=== So sánh cuối (5-fold, công bằng) ===")
    print(final_compare.to_string(index=False))

    best_row = final_compare.iloc[0]
    print(f"\nModel tốt nhất: {best_row['model']}")

    tuned_lookup = {
        "XGBoost_tuned (mới)": xgb_tuned,
        "RandomForest_tuned (mới)": rf_tuned,
        "CatBoost_tuned (mới)": cb_tuned,
    }

    if best_row["model"] in tuned_lookup:
        final_model = tuned_lookup[best_row["model"]]
        final_model.fit(X, y)
        joblib.dump(final_model, MODEL_PATH)
        model_name = best_row["model"].split(" ")[0]
        save_model_meta(META_PATH, model_name, SEED,
                        {"cv_RMSLE_5fold": float(best_row["RMSLE_5fold"])})
        print(f"Đã cập nhật {MODEL_PATH.name} bằng {model_name}.")
    else:
        print("Model đã lưu vẫn tốt nhất/gần bằng -> giữ nguyên best_model.pkl.")


if __name__ == "__main__":
    main()
