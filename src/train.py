"""
train.py
Train và so sánh nhiều model cho bài toán dự đoán giá nhà.
Chạy trực tiếp: python src/train.py
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor

from data_loader import load_processed_data

MODEL_DIR = Path("models")
TARGET_COL = "price"


def get_models() -> dict:
    """
    Linear/Ridge cần StandardScaler vì dữ liệu trộn cả biến liên tục
    quy mô lớn (sqft_living hàng nghìn) với biến nhị phân 0/1 (city
    one-hot, waterfront...) -> nếu không scale, hệ số hồi quy bị lệch
    và dễ overfit trên các cột one-hot thưa.

    RidgeCV tự tìm alpha tốt nhất bằng cross-validation thay vì cố
    định alpha=1.0, giúp regularization phù hợp hơn với 100+ chiều
    sau one-hot.

    Model dạng cây (DecisionTree/RandomForest/XGBoost) không cần scale.
    Thứ tự models cố ý đi từ đơn giản -> phức tạp để dễ so sánh:
    baseline tuyến tính -> 1 cây đơn (dễ overfit) -> ensemble (ổn định
    hơn) -> gradient boosting (thường mạnh nhất với dữ liệu dạng bảng).
    """
    return {
        "LinearRegression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LinearRegression()),
        ]),
        "RidgeCV": Pipeline([
            ("scaler", StandardScaler()),
            ("model", RidgeCV(alphas=[0.1, 1.0, 10.0, 50.0, 100.0, 300.0])),
        ]),
        "DecisionTree": DecisionTreeRegressor(
            max_depth=8, min_samples_leaf=5, random_state=42
        ),
        "RandomForest": RandomForestRegressor(
            n_estimators=300, max_depth=12, random_state=42, n_jobs=-1
        ),
        "XGBoost": XGBRegressor(
            n_estimators=400, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            random_state=42, n_jobs=-1,
        ),
    }


def cross_validate_models(X, y, n_splits=5) -> pd.DataFrame:
    """So sánh model bằng K-fold CV thay vì chỉ 1 lần train/test split.

    Một lần split có thể "may rủi" (tập test vô tình dễ hay khó hơn bình
    thường). CV chia dữ liệu thành 5 phần, mỗi phần lần lượt làm test,
    rồi lấy trung bình -> đánh giá công bằng và ổn định hơn.
    """
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    rows = []
    print(f"--- Cross-validation ({n_splits}-fold) ---")
    for name, model in get_models().items():
        # scoring âm vì sklearn quy ước "càng cao càng tốt"
        neg_mse = cross_val_score(
            model, X, y, cv=kfold, scoring="neg_mean_squared_error", n_jobs=-1
        )
        rmsle_scores = np.sqrt(-neg_mse)
        rows.append({
            "model": name,
            "CV_RMSLE_mean": rmsle_scores.mean(),
            "CV_RMSLE_std": rmsle_scores.std(),
        })
        print(f"{name:20s} RMSLE = {rmsle_scores.mean():.3f} "
              f"(+/- {rmsle_scores.std():.3f})")
    return pd.DataFrame(rows).sort_values("CV_RMSLE_mean")


def evaluate_on_holdout(X_train, X_test, y_train, y_test) -> pd.DataFrame:
    """Train trên tập train, đánh giá chi tiết trên 1 tập test giữ riêng.

    Dùng để có các con số cụ thể (MAE, MdAPE bằng USD) và để chọn ra
    model cuối cùng lưu lại -- CV ở trên chỉ để SO SÁNH model công bằng,
    không dùng CV để lưu model vì mỗi fold cho ra 1 model khác nhau.
    """
    results = []
    best_model = None
    best_rmse_log = np.inf

    for name, model in get_models().items():
        model.fit(X_train, y_train)
        preds_log = model.predict(X_test)

        # --- Metric chính: tính trên LOG-SCALE (ổn định) ---
        # Vì target được log1p trước khi train, đây chính là RMSLE.
        # KHÔNG dùng RMSE trên thang giá gốc để chọn model: chỉ vài
        # căn nhà siêu đắt bị lệch nhẹ trong log-space cũng bị expm1
        # khuếch đại thành sai số hàng chục triệu USD, làm RMSE gốc
        # trở nên vô nghĩa (từng thấy R2 âm rất sâu vì lý do này).
        rmse_log = np.sqrt(mean_squared_error(y_test, preds_log))
        r2_log = r2_score(y_test, preds_log)

        # --- Metric phụ: quy đổi ra USD để dễ hình dung, dùng MAE/MdAPE
        # (đo bằng giá trị tuyệt đối / phần trăm, không bình phương nên
        # không bị vài outlier chi phối như RMSE) ---
        preds = np.expm1(preds_log)
        y_true = np.expm1(y_test)
        mae = mean_absolute_error(y_true, preds)
        mdape = np.median(np.abs((y_true - preds) / y_true)) * 100  # %

        results.append({
            "model": name, "RMSLE": rmse_log, "R2_log": r2_log,
            "MAE_usd": mae, "MdAPE_%": mdape,
        })
        print(f"{name:20s} RMSLE={rmse_log:.3f}  R2_log={r2_log:.3f}  "
              f"MAE=${mae:,.0f}  MdAPE={mdape:.1f}%")

        if rmse_log < best_rmse_log:
            best_rmse_log = rmse_log
            best_model = model

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(best_model, MODEL_DIR / "best_model.pkl")
    print(f"\nĐã lưu model tốt nhất vào {MODEL_DIR / 'best_model.pkl'}")

    return pd.DataFrame(results).sort_values("RMSLE")


def train_and_compare():
    df = load_processed_data()

    X = df.drop(columns=[TARGET_COL])
    y = np.log1p(df[TARGET_COL])  # log-transform target

    cv_df = cross_validate_models(X, y)
    cv_df.to_csv("outputs/reports/cv_comparison.csv", index=False)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print("\n--- Đánh giá chi tiết trên tập test giữ riêng ---")
    results_df = evaluate_on_holdout(X_train, X_test, y_train, y_test)
    results_df.to_csv("outputs/reports/model_comparison.csv", index=False)

    return cv_df, results_df


if __name__ == "__main__":
    train_and_compare()
