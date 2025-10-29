from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

IN_PATH = Path("outputs/processed/apple_features.csv")
FIG_DIR = Path("outputs/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

def plot_price(df):
    plt.figure()
    df.plot(x="Date", y="Adj Close")
    plt.title("AAPL Adjusted Close (1980–2021)")
    plt.xlabel("Date"); plt.ylabel("Adj Close")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "01_adj_close.png", dpi=150)

def plot_returns(df):
    plt.figure()
    df["ret_1"].hist(bins=80)
    plt.title("Daily Returns Distribution")
    plt.xlabel("Daily Return"); plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "02_ret_hist.png", dpi=150)

def plot_ma(df):
    plt.figure()
    df.plot(x="Date", y=["Adj Close","ma_20","ma_60"])
    plt.title("AAPL Price with MA20 and MA60")
    plt.xlabel("Date"); plt.ylabel("Price")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "03_ma_cross.png", dpi=150)

def main():
    df = pd.read_csv(IN_PATH)
    df["Date"] = pd.to_datetime(df["Date"])
    plot_price(df)
    plot_returns(df)
    plot_ma(df)
    print(f"✅ Figures saved to: {FIG_DIR}")

if __name__ == "__main__":
    main()
