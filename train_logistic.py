# src/train_logistic_cv_table.py
from pathlib import Path
import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score
)
from sklearn.model_selection import TimeSeriesSplit
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["PingFang TC", "Microsoft JhengHei", "Heiti TC", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

# 前處理好的資料路徑
DATA_PATH = Path("outputs/processed/apple_features.csv")
FIG_DIR = Path("outputs/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)


def log(msg: str):
    print(f"[INFO] {msg}")


def make_dataset():
    # (這部分保持不變，讀取資料與標註風險)
    df = pd.read_csv(DATA_PATH)
    df = df.reset_index(drop=True)
    df["vol"] = df["Close"].diff().abs()
    df = df.dropna().reset_index(drop=True)
    thresh = df["vol"].quantile(0.7)
    df["risk"] = (df["vol"] >= thresh).astype(int)
    drop_cols = ["vol", "risk", "Date"]
    feature_cols = [c for c in df.columns if c not in drop_cols]
    X = df[feature_cols].to_numpy()
    y = df["risk"].to_numpy()
    n = len(df)
    split_idx = int(n * 0.8)
    X_train = X[:split_idx]
    y_train = y[:split_idx]
    X_test = X[split_idx:]
    y_test = y[split_idx:]
    return X_train, y_train, X_test, y_test, feature_cols, df


def get_single_fold_metrics(y_true, y_pred, y_prob):
    """計算單一折的各項指標"""
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1-Score": f1_score(y_true, y_pred, zero_division=0),
        "AUC": roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.5
    }


def main():
    X_train, y_train, X_test, y_test, feature_cols, df = make_dataset()

    log("開始進行 TimeSeriesSplit 五折交叉驗證（生成詳細表格） ...")
    
    tscv = TimeSeriesSplit(n_splits=5)
    
    # 用來儲存每一折的詳細數據 (Row Data)
    val_table_rows = []
    train_table_rows = []
    
    # 用來計算累積平均 (Cumulative Stats)
    acc_history_val = []
    acc_history_train = []

    fold = 0

    for tr_idx, val_idx in tscv.split(X_train):
        fold += 1
        X_tr, X_val = X_train[tr_idx], X_train[val_idx]
        y_tr, y_val = y_train[tr_idx], y_train[val_idx]

        # 訓練模型
        model = LogisticRegression(max_iter=1000)
        model.fit(X_tr, y_tr)

        # ==========================
        # 1. 驗證集 (Test on Fold)
        # ==========================
        y_val_pred = model.predict(X_val)
        y_val_prob = model.predict_proba(X_val)[:, 1]
        
        # 計算指標
        m_val = get_single_fold_metrics(y_val, y_val_pred, y_val_prob)
        
        # 計算累積 Mean ± Std (針對 Accuracy)
        acc_history_val.append(m_val["Accuracy"])
        cum_mean_val = np.mean(acc_history_val)
        cum_std_val = np.std(acc_history_val)
        
        # 建立 Row
        val_row = {
            "Pass": fold,
            "N_Test_Windows": len(y_val),
            "Accuracy": m_val["Accuracy"],
            "Precision": m_val["Precision"],
            "Recall": m_val["Recall"],
            "F1-Score": m_val["F1-Score"],
            "AUC": m_val["AUC"],
            "Mean ± Std Dev": f"{cum_mean_val:.4f} ± {cum_std_val:.4f}"
        }
        val_table_rows.append(val_row)

        # ==========================
        # 2. 訓練集 (Train on Fold)
        # ==========================
        y_tr_pred = model.predict(X_tr)
        y_tr_prob = model.predict_proba(X_tr)[:, 1]
        
        m_tr = get_single_fold_metrics(y_tr, y_tr_pred, y_tr_prob)
        
        # 計算累積 Mean ± Std (針對 Accuracy)
        acc_history_train.append(m_tr["Accuracy"])
        cum_mean_tr = np.mean(acc_history_train)
        cum_std_tr = np.std(acc_history_train)
        
        train_row = {
            "Pass": fold,
            "N_Train_Windows": len(y_tr),
            "Accuracy": m_tr["Accuracy"],
            "Precision": m_tr["Precision"],
            "Recall": m_tr["Recall"],
            "F1-Score": m_tr["F1-Score"],
            "AUC": m_tr["AUC"],
            "Mean ± Std Dev": f"{cum_mean_tr:.4f} ± {cum_std_tr:.4f}"
        }
        train_table_rows.append(train_row)

    # ==========================
    # 輸出漂亮的表格
    # ==========================
    
    # 定義顯示格式
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    pd.set_option('display.unicode.east_asian_width', True) # 對齊中文字寬

    # 1. 測試集結果 (Validation Set Results)
    df_val = pd.DataFrame(val_table_rows)
    print("\n" + "="*30)
    print(" 測試集結果 (Validation Set)")
    print("="*30)
    print(df_val.to_string(index=False, float_format="%.4f"))

    # 2. 訓練集結果 (Training Set Results)
    df_train = pd.DataFrame(train_table_rows)
    print("\n" + "="*30)
    print(" 訓練集結果 (Training Set)")
    print("="*30)
    print(df_train.to_string(index=False, float_format="%.4f"))
    
    # 3. 匯出成 Excel 或 CSV (選用)
    # df_val.to_csv("outputs/cv_validation_results.csv", index=False)
    # df_train.to_csv("outputs/cv_training_results.csv", index=False)

    print("\n[INFO] Mean ± Std Dev 欄位代表截至該 Pass 為止的「累積」平均準確率與標準差。")

    # (原本後面的 最終訓練與畫圖程式碼 可保留或省略) 
    # ...

if __name__ == "__main__":
    main()