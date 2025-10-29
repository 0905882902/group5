from pathlib import Path
import pandas as pd
import numpy as np

# ------- 路徑設定 -------
RAW_PATH = Path("data/raw.csv")
OUT_DIR  = Path("outputs/processed")
OUT_PATH = OUT_DIR / "apple_features.csv"

# 安全建立資料夾（存在就略過）
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ------- 讀檔（若檔名不同會即時提示）-------
if not RAW_PATH.exists() or RAW_PATH.stat().st_size == 0:
    # 自動改讀 data/ 裡最大的一個 csv，避免放錯檔名
    csvs = sorted(Path("data").glob("*.csv"),
                  key=lambda p: p.stat().st_size if p.exists() else 0,
                  reverse=True)
    if not csvs:
        raise FileNotFoundError("找不到可用的 CSV，請把原始檔放到 data/ 並命名為 raw.csv")
    print(f"⚠️ raw.csv 無效，改讀：{csvs[0].name}")
    RAW_PATH = csvs[0]

print(f"📖 讀取檔案：{RAW_PATH.resolve()}")
df = pd.read_csv(RAW_PATH)

# ------- 基礎清理 -------
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date").drop_duplicates(subset=["Date"])
df = df[(df["High"] >= df["Low"]) & (df["Volume"] > 0)]
df = df.ffill()

# ------- 特徵工程 -------
s = df["Adj Close"]
df["ret_1"]  = s.pct_change(1)
df["ret_5"]  = s.pct_change(5)
df["ret_20"] = s.pct_change(20)

for w in [5, 20, 60]:
    df[f"ma_{w}"] = s.rolling(w).mean()
df["bias_20"] = (s - df["ma_20"]) / df["ma_20"]
df["vol_20"]  = df["ret_1"].rolling(20).std()
df["vol_chg"] = df["Volume"].pct_change(1)

# RSI
delta = s.diff()
gain  = delta.clip(lower=0).rolling(14).mean()
loss  = (-delta.clip(upper=0)).rolling(14).mean()
rs = gain / loss
df["rsi_14"] = 100 - (100 / (1 + rs))

# MACD
ema12 = s.ewm(span=12, adjust=False).mean()
ema26 = s.ewm(span=26, adjust=False).mean()
df["macd"]     = ema12 - ema26
df["macd_sig"] = df["macd"].ewm(span=9, adjust=False).mean()
df["macd_hist"]= df["macd"] - df["macd_sig"]

# 標籤（明日漲跌）
df["y_updown"] = (s.shift(-1) > s).astype(int)

# 移除因 rolling/shift 產生的 NA
df = df.dropna().reset_index(drop=True)

# ------- 輸出 -------
print(f"🗂 目標輸出路徑：{OUT_PATH.resolve()}")
df.to_csv(OUT_PATH, index=False)
print(f"✅ Saved: {OUT_PATH}  (rows={df.shape[0]}, cols={df.shape[1]})")
