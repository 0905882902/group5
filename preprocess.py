# src/preprocess.py
from pathlib import Path
import pandas as pd
import numpy as np

# 可能的原始檔案（你有一個中文檔名的我也放進來）
INPUT_FILES = [
    Path("data/raw.csv"),
    Path("data/1980 年至 2021 年的蘋果股票價格.csv"),
    Path("data/1980年至2021年的蘋果股票價格.csv"),
]

# 輸出位置
OUT_DIR = Path("outputs/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = OUT_DIR / "apple_features.csv"

# 如果你的資料有「標籤/類別欄位」，在這裡寫名字；沒有就設 None
TARGET_COL = None    # 例：TARGET_COL = "label"


def log(msg: str):
    print(f"[INFO] {msg}")


# 1. 讀資料 -------------------------------------------------------
def load_data() -> pd.DataFrame:
    for p in INPUT_FILES:
        if p.exists():
            df = pd.read_csv(p)
            log(f"資料載入完成：{df.shape[0]} 列、{df.shape[1]} 欄，來源：{p}")
            return df
    raise FileNotFoundError("找不到可用的原始資料，請確認 CSV 放在 data/ 底下。")


# 2. 缺失值處理 ---------------------------------------------------
def handle_missing(df: pd.DataFrame) -> pd.DataFrame:
    log("開始處理缺失值 ...")
    df = df.copy()

    num_cols = df.select_dtypes(include=np.number).columns
    obj_cols = df.select_dtypes(include="object").columns

    # 數值欄位：用中位數補
    for col in num_cols:
        df[col] = df[col].fillna(df[col].median())

    # 字串欄位：用 "Unknown" 補
    for col in obj_cols:
        df[col] = df[col].fillna("Unknown")

    log("缺失值處理完成")
    return df


# 3. 異常值處理 (用 Z-score 刪掉極端值) ---------------------------
def remove_outliers(df: pd.DataFrame, z_thresh: float = 3.0) -> pd.DataFrame:
    log("開始移除異常值 ...")
    df = df.copy()
    before = df.shape[0]

    num_cols = df.select_dtypes(include=np.number).columns
    for col in num_cols:
        mean = df[col].mean()
        std = df[col].std()
        if std == 0 or np.isnan(std):
            # 標準差 0 就沒辦法算 z-score，跳過
            continue
        z = (df[col] - mean) / std
        df = df[np.abs(z) < z_thresh]

    log(f"異常值處理完成，剩下 {df.shape[0]} 列（原本 {before} 列）")
    return df


# 4. 類別編碼（不用 sklearn 的 LabelEncoder，直接用 pandas 的 category） ---
def encode_categorical(df: pd.DataFrame) -> pd.DataFrame:
    log("開始做類別編碼 ...")
    df = df.copy()
    obj_cols = df.select_dtypes(include="object").columns

    for col in obj_cols:
        df[col] = df[col].astype("category").cat.codes  # 會轉成 0,1,2,...
    log("類別編碼完成")
    return df


# 5. 數值縮放（標準化） ------------------------------------------
def scale_numeric(df: pd.DataFrame, method: str = "standard") -> pd.DataFrame:
    log("開始做數值縮放 ...")
    df = df.copy()
    num_cols = df.select_dtypes(include=np.number).columns

    if method == "standard":
        # (x - mean) / std
        for col in num_cols:
            mean = df[col].mean()
            std = df[col].std()
            if std == 0 or np.isnan(std):
                df[col] = 0.0
            else:
                df[col] = (df[col] - mean) / std
    else:
        # min-max
        for col in num_cols:
            mn = df[col].min()
            mx = df[col].max()
            if mx == mn:
                df[col] = 0.0
            else:
                df[col] = (df[col] - mn) / (mx - mn)

    log("數值縮放完成")
    return df


# 6. PCA 降維（純 numpy） ---------------------------------------
def add_pca(df: pd.DataFrame, n_components: int = 2) -> pd.DataFrame:
    log("開始做 PCA 降維 ...")
    df = df.copy()
    num_cols = df.select_dtypes(include=np.number).columns

    X = df[num_cols].to_numpy().astype(float)

    # 中心化
    X_centered = X - X.mean(axis=0)

    # 共變異矩陣
    cov = np.cov(X_centered, rowvar=False)

    # 特徵分解
    eigvals, eigvecs = np.linalg.eigh(cov)

    # 大到小排序
    idx = np.argsort(eigvals)[::-1]
    eigvecs = eigvecs[:, idx]

    # 取前 n_components
    eigvecs = eigvecs[:, :n_components]

    # 投影
    pcs = np.dot(X_centered, eigvecs)

    for i in range(n_components):
        df[f"PC{i+1}"] = pcs[:, i]

    log("PCA 降維完成，已新增 PC 欄位")
    return df


# 7. 資料擴增與類別平衡（random oversampling） --------------------
def random_oversample(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """
    不用 imbalanced-learn 自己做的 oversampling：
    把每一類補到跟最多的那一類一樣多。
    """
    log("開始做資料擴增 / 類別平衡（Random Oversampling） ...")
    df = df.copy()

    counts = df[target_col].value_counts()
    max_count = counts.max()

    rng = np.random.default_rng(42)
    parts = []

    for cls, cnt in counts.items():
        df_cls = df[df[target_col] == cls]
        if cnt < max_count:
            need = max_count - cnt
            # 隨機抽樣補
            extra_idx = rng.integers(low=0, high=cnt, size=need)
            extra = df_cls.iloc[extra_idx]
            df_cls = pd.concat([df_cls, extra], axis=0)
        parts.append(df_cls)

    df_balanced = pd.concat(parts, axis=0).sample(frac=1.0, random_state=42).reset_index(drop=True)
    log(f"資料擴增 / 類別平衡完成：現在共有 {df_balanced.shape[0]} 列")
    return df_balanced


# 主流程 -----------------------------------------------------------
def main():
    # 讀資料
    df = load_data()
    # 缺失值
    df = handle_missing(df)
    # 異常值
    df = remove_outliers(df)
    # 編碼
    df = encode_categorical(df)
    # 歸一化
    df = scale_numeric(df, method="standard")
    # 降維
    df = add_pca(df, n_components=2)
    # 擴增 & 類別平衡（有設定才做）
    if TARGET_COL is not None and TARGET_COL in df.columns:
        df = random_oversample(df, TARGET_COL)
    else:
        log("未設定 TARGET_COL，跳過資料擴增 / 類別平衡")

    # 輸出
    df.to_csv(OUT_PATH, index=False)
    log(f"✅ 前處理完成，已輸出到：{OUT_PATH}")


if __name__ == "__main__":
    main()

