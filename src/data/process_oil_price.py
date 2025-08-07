import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

def main():
    """
    1. 결측치 보간, 주말 및 공휴일 전날 값으로 채우기
    2. 이상치 탐지 (7일 window 기준 Z-score > 3)
    3. spike 판단 (하루 등락률 ≥ 5% OR 3일 누적 등락률 ≥ 10% OR 7일 누적 등락률 ≥ 10%)
    4. oil_spike, oil_rise, oil_fall 컬럼 추가
    5. oil_price_processed 파일 저장
    6. 시각화
    """
    # 데이터 로드
    data_dir = Path(__file__).parents[2] / "data"
    oil = pd.read_csv(data_dir/"oil_price.csv", parse_dates=["date"])
    
    # 결측 보간
    missing = oil.isna().sum()          
    print("=== Missing Values ===")
    print(missing, "\n")

    oil = oil.set_index("date").sort_index()
    full_idx = pd.date_range(oil.index.min(), oil.index.max(), freq="D")
    oil = oil.reindex(full_idx)
    oil['brent_usd'] = oil['brent_usd'].ffill()
    oil.index.name = "date"

    # 이상치 탐지
    window = 7 
    rolling_mean = oil['brent_usd'].rolling(window=window, center=True).mean()
    rolling_std  = oil['brent_usd'].rolling(window=window, center=True).std()
    oil['z_score']   = (oil['brent_usd'] - rolling_mean) / rolling_std
    oil['z_outlier'] = oil['z_score'].abs() > 3

    print("=== Z-Score Outlier Detection ===")
    print("Total days:", len(oil))
    print("Z-Score outliers flagged:", int(oil['z_outlier'].sum()), "rows\n")
    print(oil[oil['z_outlier']].head(), "\n")

    # spike 및 rise/fall 판단
    oil["abs_chg_1d"] = oil["brent_usd"].diff()
    oil["abs_chg_3d"] = oil["brent_usd"] - oil["brent_usd"].shift(3)
    oil["abs_chg_7d"] = oil["brent_usd"] - oil["brent_usd"].shift(7)
    oil["pct_chg_7d"] = oil["brent_usd"] / oil["brent_usd"].shift(7) - 1
    oil["pct_chg_1d"] = oil["brent_usd"].pct_change()
    oil["pct_chg_3d"] = oil["brent_usd"] / oil["brent_usd"].shift(3) - 1

    cond_1d = (oil["pct_chg_1d"].abs() >= 0.05) & (oil["abs_chg_1d"].abs() >= 5)
    cond_3d = (oil["pct_chg_3d"].abs() >= 0.08) & (oil["abs_chg_3d"].abs() >= 5)
    cond_7d = (oil["pct_chg_7d"].abs() >= 0.10) & (oil["abs_chg_7d"].abs() >= 10)

    oil["oil_spike"] = cond_1d | cond_3d | cond_7d

    oil["oil_rise"] = (
    ((oil["pct_chg_1d"] >= 0.05) & (oil["abs_chg_1d"] >= 5)) |
    ((oil["pct_chg_3d"] >= 0.08) & (oil["abs_chg_3d"] >= 5)) |
    ((oil["pct_chg_7d"] >= 0.10) & (oil["abs_chg_7d"] >= 10))
    )

    oil["oil_fall"] = (
        ((oil["pct_chg_1d"] <= -0.05) & (oil["abs_chg_1d"] <= -5)) |
        ((oil["pct_chg_3d"] <= -0.08) & (oil["abs_chg_3d"] <= -5)) |
        ((oil["pct_chg_7d"] <= -0.10) & (oil["abs_chg_7d"] <= -10))
    )

    print("After processing (hybrid criteria):")
    print("Total days:", len(oil))
    print("Spike events:", int(oil["oil_spike"].sum()))
    print(f'Spike ratio: {(int(oil["oil_spike"].sum())/len(oil))*100:.2f}%')
    print("")

    # 시각화
    plt.figure(figsize=(15, 6))
    plt.plot(oil.index, oil["brent_usd"], label="Brent Oil Price (USD)", color="blue")

    spike_dates = oil[oil["oil_spike"]].index
    spike_values = oil.loc[spike_dates, "brent_usd"]
    plt.scatter(spike_dates, spike_values, color="red", label="Hybrid Spike Event", zorder=5)

    plt.title("Brent Oil Price with Hybrid Spike Events (1-day ≥5% or 3-day ≥10%)")
    plt.xlabel("Date")
    plt.ylabel("Price (USD)")
    plt.legend()
    plt.tight_layout()
    plt.grid(True)
    plt.show()

    # 전처리된 파일 저장

    # (1) 불필요한 컬럼 제거
    cols_to_drop = [
        "z_score", "z_outlier",
        "pct_chg_1d", "pct_chg_3d", "pct_chg_7d",
        "abs_chg_1d", "abs_chg_3d", "abs_chg_7d"
    ]
    oil.drop(columns=cols_to_drop, inplace=True)

    # (2) 불리언 → 정수 변환
    oil["oil_spike"] = oil["oil_spike"].astype(int)
    oil["oil_rise"]  = oil["oil_rise"].astype(int)
    oil["oil_fall"]  = oil["oil_fall"].astype(int)
    
    out_path = data_dir/"oil_price_processed.csv"
    oil.reset_index().to_csv(out_path, index=False, date_format="%Y-%m-%d")
    print(f"Saved processed data to {out_path}")

if __name__ == "__main__":
    main()
