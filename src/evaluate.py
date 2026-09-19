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
from utils import get_final_estimator

MODEL_PATH = Path("models/best_model.pkl")
FIG_DIR = Path("outputs/figures")


REPORT_DIR = Path("outputs/reports")


def plot_predicted_vs_actual(y_true, y_pred):
    """Vẽ trên trục log-log: vài căn nhà siêu đắt sẽ không còn 'nuốt chửng'
    phần còn lại của biểu đồ như khi vẽ trên thang USD gốc."""
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.3, s=10)
    plt.xscale("log")
    plt.yscale("log")
    lims = [y_true.min(), y_true.max()]
    plt.plot(lims, lims, "r--")
    plt.xlabel("Giá thực tế (log scale)")
    plt.ylabel("Giá dự đoán (log scale)")
    plt.title("Predicted vs Actual")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "predicted_vs_actual.png", dpi=110)


def plot_residuals(y_test_log, preds_log):
    """Phân tích residual TRÊN LOG-SCALE (không phải USD).

    Lý do dùng log-scale: residual = y_true - y_pred trên thang USD sẽ
    tự động lớn hơn nhiều với nhà đắt (một sai số 10% của nhà 2 triệu
    là 200k, trong khi 10% của nhà 300k chỉ là 30k) -> biểu đồ residual
    trên USD sẽ trông như "phễu" (heteroscedastic) dù model không thực
    sự tệ hơn ở phân khúc giá cao. Residual trên log-scale xấp xỉ với
    sai số phần trăm, cho phép so sánh công bằng giữa các phân khúc giá.
    """
    residuals = y_test_log - preds_log

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # (a) Residual vs Predicted -> tìm pattern hệ thống (lý tưởng: đám mây
    # ngẫu nhiên quanh 0, không có hình dạng cong hay phễu)
    axes[0].scatter(preds_log, residuals, alpha=0.3, s=10)
    axes[0].axhline(0, color="red", linestyle="--")
    axes[0].set_xlabel("Giá trị dự đoán (log scale)")
    axes[0].set_ylabel("Residual (log thực tế - log dự đoán)")
    axes[0].set_title("Residual vs Predicted")

    # (b) Phân phối residual -> lý tưởng: giống phân phối chuẩn, tâm ở 0
    axes[1].hist(residuals, bins=50, color="teal")
    axes[1].axvline(0, color="red", linestyle="--")
    axes[1].set_xlabel("Residual (log scale)")
    axes[1].set_title("Phân phối Residual")

    plt.tight_layout()
    plt.savefig(FIG_DIR / "residual_analysis.png", dpi=110)
    return residuals


def error_by_price_segment(y_true_usd, preds_usd, n_bins=5) -> pd.DataFrame:
    """Chia nhà thành các phân khúc giá (VD: rẻ -> đắt) và tính sai số
    riêng cho từng phân khúc. Giúp trả lời câu hỏi: model có dự đoán tệ
    hơn hẳn ở nhà siêu rẻ hay siêu đắt không, hay đồng đều?"""
    df = pd.DataFrame({"y_true": y_true_usd, "y_pred": preds_usd})
    df["abs_pct_error"] = np.abs(df["y_true"] - df["y_pred"]) / df["y_true"]

    df["price_segment"] = pd.qcut(
        df["y_true"], q=n_bins,
        labels=[f"Q{i+1}" for i in range(n_bins)]
    )

    summary = df.groupby("price_segment", observed=True).agg(
        n=("y_true", "size"),
        price_range_min=("y_true", "min"),
        price_range_max=("y_true", "max"),
        MdAPE=("abs_pct_error", "median"),
        MAE=("y_true", lambda s: np.mean(
            np.abs(s - df.loc[s.index, "y_pred"])
        )),
    ).reset_index()
    summary["MdAPE"] = (summary["MdAPE"] * 100).round(1)
    summary["MAE"] = summary["MAE"].round(0)
    return summary


def worst_predictions(y_true_usd, preds_usd, X_test, top_n=10) -> pd.DataFrame:
    """Liệt kê top N căn nhà bị dự đoán sai lệch nhiều nhất, kèm vài đặc
    điểm chính -- hữu ích để tìm xem model yếu ở loại nhà nào (VD: nhà
    có waterfront, view cao, hoặc thành phố hiếm ít dữ liệu)."""
    df = pd.DataFrame({
        "price_true": y_true_usd.values,
        "price_pred": preds_usd,
    }, index=y_true_usd.index)
    df["abs_error"] = np.abs(df["price_true"] - df["price_pred"])
    df["pct_error"] = (df["abs_error"] / df["price_true"] * 100).round(1)

    cols_of_interest = ["sqft_living", "bedrooms", "bathrooms", "view", "waterfront"]
    cols_of_interest = [c for c in cols_of_interest if c in X_test.columns]
    df = df.join(X_test[cols_of_interest])

    return df.sort_values("abs_error", ascending=False).head(top_n)


def plot_feature_importance(model, feature_names, top_n=15):
    estimator = get_final_estimator(model)

    if hasattr(estimator, "feature_importances_"):
        values = estimator.feature_importances_
    elif hasattr(estimator, "coef_"):
        # Với Linear/Ridge: dùng trị tuyệt đối hệ số (đã scale) làm "importance"
        values = np.abs(estimator.coef_)
    else:
        print("Model này không hỗ trợ feature importance.")
        return

    importances = pd.Series(values, index=feature_names)
    importances = importances.sort_values(ascending=False).head(top_n)

    plt.figure(figsize=(8, 6))
    importances.sort_values().plot(kind="barh")
    plt.title(f"Top {top_n} Feature Importance")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "feature_importance.png", dpi=110)


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_processed_data()

    X = df.drop(columns=["price"])
    y = np.log1p(df["price"])

    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = joblib.load(MODEL_PATH)
    preds_log = model.predict(X_test)

    y_true = np.expm1(y_test)
    preds = np.expm1(preds_log)

    plot_predicted_vs_actual(y_true, preds)
    plot_residuals(y_test, preds_log)
    plot_feature_importance(model, X.columns)

    segment_df = error_by_price_segment(y_true, preds)
    segment_df.to_csv(REPORT_DIR / "error_by_price_segment.csv", index=False)
    print("\n=== Sai số theo phân khúc giá (Q1=rẻ nhất -> Q5=đắt nhất) ===")
    print(segment_df.to_string(index=False))

    worst_df = worst_predictions(y_true, preds, X_test)
    worst_df.to_csv(REPORT_DIR / "worst_predictions.csv", index=False)
    print(f"\n=== Top {len(worst_df)} dự đoán sai lệch nhiều nhất ===")
    print(worst_df.to_string(index=False))

    print(f"\nĐã lưu biểu đồ vào {FIG_DIR}/ và báo cáo vào {REPORT_DIR}/")


if __name__ == "__main__":
    main()
