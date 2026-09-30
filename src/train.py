"""
train.py
Train và so sánh nhiều model cho bài toán dự đoán giá nhà.
Chạy trực tiếp: python src/train.py
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.base import clone
from sklearn.ensemble import (
    RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor,
    StackingRegressor,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

from config import CFG, resolve_path
from data_loader import load_processed_data
from utils import save_model_meta

MODEL_PATH = resolve_path(CFG["paths"]["model"])
META_PATH = resolve_path(CFG["paths"]["model_meta"])
REPORT_DIR = resolve_path(CFG["paths"]["reports_dir"])
TARGET_COL = CFG["data"]["target"]
SEED = CFG["seed"]


def build_base_models(cfg: dict = None) -> dict:
    """8 model đơn, cố ý sắp xếp từ đơn giản -> phức tạp để dễ so sánh
    (siêu tham số lấy từ config.yaml, mục `models`):

    1. LinearRegression  - baseline tuyến tính, không regularize
    2. RidgeCV            - tuyến tính + L2 regularization, tự động chọn
                             alpha tốt nhất bằng cross-validation nội bộ
                             (thay vì cố định 1 giá trị) -> phù hợp hơn
                             với 43 chiều sau one-hot city
    3. DecisionTree       - 1 cây đơn, dễ hiểu nhưng dễ overfit
    4. RandomForest       - bagging nhiều cây (giảm variance)
    5. ExtraTrees         - giống RandomForest nhưng chọn ngưỡng chia
                             nhánh ngẫu nhiên hơn -> train nhanh hơn,
                             thường variance thấp hơn, bias cao hơn 1 chút
    6. GradientBoosting   - boosting tuần tự (sklearn, CPU, không quá
                             nhanh nhưng là nền tảng ý tưởng của XGBoost)
    7. XGBoost            - gradient boosting tối ưu hiệu năng, thường
                             mạnh nhất với dữ liệu dạng bảng
    8. CatBoost           - gradient boosting xử lý tốt biến hạng mục,
                             ít cần tinh chỉnh mà vẫn cho kết quả tốt

    Linear/Ridge cần StandardScaler vì dữ liệu trộn cả biến liên tục quy
    mô lớn (sqft_living hàng nghìn) với biến nhị phân 0/1 (city one-hot,
    waterfront...) -> nếu không scale, hệ số hồi quy bị lệch và dễ
    overfit trên các cột one-hot thưa. Mọi model dạng cây/boosting không
    cần scale.
    """
    cfg = cfg or CFG
    m, seed, n_jobs = cfg["models"], cfg["seed"], cfg["n_jobs"]
    return {
        "LinearRegression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LinearRegression()),
        ]),
        "RidgeCV": Pipeline([
            ("scaler", StandardScaler()),
            ("model", RidgeCV(alphas=m["ridge"]["alphas"])),
        ]),
        "DecisionTree": DecisionTreeRegressor(
            random_state=seed, **m["decision_tree"]
        ),
        "RandomForest": RandomForestRegressor(
            random_state=seed, n_jobs=n_jobs, **m["random_forest"]
        ),
        "ExtraTrees": ExtraTreesRegressor(
            random_state=seed, n_jobs=n_jobs, **m["extra_trees"]
        ),
        "GradientBoosting": GradientBoostingRegressor(
            random_state=seed, **m["gradient_boosting"]
        ),
        "XGBoost": XGBRegressor(
            random_state=seed, n_jobs=n_jobs, **m["xgboost"]
        ),
        "CatBoost": CatBoostRegressor(
            random_state=seed, verbose=False, **m["catboost"]
        ),
    }


def build_stacking(base_models: dict, cfg: dict = None) -> StackingRegressor:
    """Stacking ensemble: model thứ 9.

    Ý tưởng: mỗi base model học riêng, rồi 1 model "cấp 2" (RidgeCV) học
    cách trộn dự đoán của chúng. Dự đoán đưa cho model cấp 2 là dự đoán
    NGOÀI-FOLD (cross_val_predict bên trong StackingRegressor), nên không
    bị rò rỉ: base model chưa từng thấy dòng mà nó dự đoán.

    Vì sao mặc định chọn RidgeCV + ExtraTrees + XGBoost + CatBoost: stacking
    chỉ có lợi khi các base model SAI KHÁC NHAU. Ridge (tuyến tính), cây
    bagging (ExtraTrees) và 2 boosting cho 3 kiểu lỗi khác nhau; trộn
    thêm DecisionTree/RandomForest chỉ tốn thời gian vì rất giống ExtraTrees.
    Danh sách này đổi được trong config.yaml (stacking.base_models).

    Chi phí: mỗi lần fit huấn luyện mỗi base model (cv_splits + 1) lần, và
    cross_validate_models còn bọc thêm 5-fold bên ngoài -> chậm hơn model
    đơn khoảng (cv_splits + 1) lần. Tắt bằng stacking.enabled: false.
    """
    cfg = cfg or CFG
    scfg = cfg["stacking"]
    names = scfg["base_models"]

    unknown = [n for n in names if n not in base_models]
    if unknown:
        raise ValueError(
            f"stacking.base_models có tên không hợp lệ: {unknown}. "
            f"Chọn trong: {sorted(base_models)}"
        )
    if len(set(names)) != len(names) or len(names) < 2:
        raise ValueError("stacking.base_models cần >= 2 tên khác nhau.")

    return StackingRegressor(
        estimators=[(n, clone(base_models[n])) for n in names],
        final_estimator=RidgeCV(alphas=scfg["final_alphas"]),
        cv=KFold(n_splits=scfg["cv_splits"], shuffle=True,
                 random_state=cfg["seed"]),
        passthrough=scfg["passthrough"],
        n_jobs=1,  # base model (XGB, ET...) đã tự song song bên trong
    )


def get_models(cfg: dict = None) -> dict:
    """8 model đơn + (nếu stacking.enabled) Stacking ở cuối."""
    cfg = cfg or CFG
    models = build_base_models(cfg)
    if cfg["stacking"]["enabled"]:
        models["Stacking"] = build_stacking(models, cfg)
    return models


def cross_validate_models(X, y, n_splits=None) -> pd.DataFrame:
    """So sánh model bằng K-fold CV thay vì chỉ 1 lần train/test split.

    Một lần split có thể "may rủi" (tập test vô tình dễ hay khó hơn bình
    thường). CV chia dữ liệu thành 5 phần, mỗi phần lần lượt làm test,
    rồi lấy trung bình -> đánh giá công bằng và ổn định hơn.
    """
    n_splits = n_splits or CFG["cv"]["n_splits"]
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=SEED)
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
    best_name = None
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
            best_name = name

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, MODEL_PATH)
    save_model_meta(META_PATH, best_name, SEED, {"holdout_RMSLE": best_rmse_log})
    print(f"\nĐã lưu model tốt nhất ({best_name}) vào {MODEL_PATH}")

    return pd.DataFrame(results).sort_values("RMSLE")


def train_and_compare():
    df = load_processed_data()

    X = df.drop(columns=[TARGET_COL])
    y = np.log1p(df[TARGET_COL])  # log-transform target

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    cv_df = cross_validate_models(X, y)
    cv_df.to_csv(REPORT_DIR / "cv_comparison.csv", index=False)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=CFG["data"]["test_size"], random_state=SEED
    )
    print("\n--- Đánh giá chi tiết trên tập test giữ riêng ---")
    results_df = evaluate_on_holdout(X_train, X_test, y_train, y_test)
    results_df.to_csv(REPORT_DIR / "model_comparison.csv", index=False)

    return cv_df, results_df


if __name__ == "__main__":
    train_and_compare()
