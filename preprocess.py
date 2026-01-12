# src/preprocess.py
from pathlib import Path
import pandas as pd
import numpy as np

# 可能的原始檔案
INPUT_FILES = [
    Path("data/raw.csv"),
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
            continue
        z = (df[col] - mean) / std
        df = df[np.abs(z) < z_thresh]
    log(f"異常值處理完成，剩下 {df.shape[0]} 列（原本 {before} 列）")
    return df


# 4. 類別編碼 -----------------------------------------------------
def encode_categorical(df: pd.DataFrame) -> pd.DataFrame:
    log("開始做類別編碼 ...")
    df = df.copy()
    obj_cols = df.select_dtypes(include="object").columns

    for col in obj_cols:
        df[col] = df[col].astype("category").cat.codes
    log("類別編碼完成")
    return df


# 5. 數值縮放 ------------------------------------------
def scale_numeric(df: pd.DataFrame, method: str = "standard") -> pd.DataFrame:
    log("開始做數值縮放 ...")
    df = df.copy()
    num_cols = df.select_dtypes(include=np.number).columns
    if method == "standard":
        for col in num_cols:
            mean = df[col].mean()
            std = df[col].std()
            if std == 0 or np.isnan(std):
                df[col] = 0.0
            else:
                df[col] = (df[col] - mean) / std
    else:
        for col in num_cols:
            mn = df[col].min()
            mx = df[col].max()
            if mx == mn:
                df[col] = 0.0
            else:
                df[col] = (df[col] - mn) / (mx - mn)
    log("數值縮放完成")
    return df


# 6. PCoA 降維 -------------------------------
def add_pcoa(df: pd.DataFrame, n_components: int = 2) -> pd.DataFrame:
    """
    Classical MDS / PCoA (Principal Coordinates Analysis)
    使用歐氏距離 → 雙中心化 → 特徵分解
    """
    log("開始做 PCoA 降維 ...")
    df = df.copy()
    num_cols = df.select_dtypes(include=np.number).columns
    X = df[num_cols].to_numpy().astype(float)
    n = X.shape[0]
    # 6-1：建立距離平方矩陣 D^2
    row_sq = np.sum(X ** 2, axis=1)
    D2 = row_sq[:, None] + row_sq[None, :] - 2.0 * np.dot(X, X.T)
    D2[D2 < 0] = 0.0  # 防止浮點誤差
    # 6-2：雙中心化
    I = np.eye(n)
    One = np.ones((n, n)) / n
    J = I - One
    B = -0.5 * J.dot(D2).dot(J)
    # 6-3：特徵分解
    eigvals, eigvecs = np.linalg.eigh(B)
    # 由大到小排序
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    # 6-4：取前幾個正特徵值生成 PCo 坐標
    added = 0
    for i in range(min(n_components, len(eigvals))):
        lam = eigvals[i]
        if lam <= 0:
            break
        coord = eigvecs[:, i] * np.sqrt(lam)
        df[f"PCo{i+1}"] = coord
        added += 1
    log(f"PCoA 降維完成：新增 {added} 個 PCo 座標欄位")
    return df


# 7. 資料擴增與類別平衡（random oversampling） --------------------
def random_oversample(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    log("開始做資料擴增 / 類別平衡 ...")
    df = df.copy()

    counts = df[target_col].value_counts()
    max_count = counts.max()

    rng = np.random.default_rng(42)
    parts = []

    for cls, cnt in counts.items():
        df_cls = df[df[target_col] == cls]
        if cnt < max_count:
            need = max_count - cnt
            extra_idx = rng.integers(low=0, high=cnt, size=need)
            extra = df_cls.iloc[extra_idx]
            df_cls = pd.concat([df_cls, extra], axis=0)
        parts.append(df_cls)

    df_balanced = pd.concat(parts, axis=0).sample(frac=1.0, random_state=42).reset_index(drop=True)
    log(f"資料擴增 / 類別平衡完成：{df_balanced.shape[0]} 列")
    return df_balanced


# 主流程 -----------------------------------------------------------
def main():
    # 讀資料
    df = load_data()

    # ⭐ 取 2000–2009 十年資料
    df["Date"] = pd.to_datetime(df["Date"])
    df = df[(df["Date"] >= "2000-01-01") & (df["Date"] <= "2009-12-31")]

    # 缺失值
    df = handle_missing(df)

    # 異常值
    df = remove_outliers(df)

    # 類別編碼
    df = encode_categorical(df)

    # 數值縮放
    df = scale_numeric(df, method="standard")

    # ⭐ 使用 PCoA 降維
    df = add_pcoa(df, n_components=2)

    # oversampling（如 target 有設定才做）
    if TARGET_COL is not None and TARGET_COL in df.columns:
        df = random_oversample(df, TARGET_COL)
    else:
        log("未設定 TARGET_COL，跳過資料擴增 / 類別平衡")

    # 輸出
    df.to_csv(OUT_PATH, index=False)
    log(f"✅ 前處理完成，已輸出到：{OUT_PATH}")


if __name__ == "__main__":
    main()