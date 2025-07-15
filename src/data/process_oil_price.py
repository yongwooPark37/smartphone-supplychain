import pandas as pd
from pathlib import Path

def main():
    """
    1) oil_price.csv 로드
    2) 결측치 확인 및 주말·공휴일 결측 보간
    3) 이상치 탐지 (7-days window 기준 |Z-Score| > 3이면 이상치로 간주)
    4) 7일 전 대비 등락률 계산 및 이벤트 플래그
    5) 요약 및 저장 (oil_price_processed.csv)
    """
    
    # 1) 데이터 로드
    data_dir = Path(__file__).parents[2] / "data"
    oil = pd.read_csv(data_dir/"oil_price.csv", parse_dates=["date"])
    
    # 2) 결측치 탐지 및 보간
    missing = oil.isna().sum()          
    print("=== Missing Values ===")
    print(missing, "\n")

    oil = oil.set_index("date").sort_index()
    full_idx = pd.date_range(oil.index.min(), oil.index.max(), freq="D")
    oil = oil.reindex(full_idx)
    oil['brent_usd'] = oil['brent_usd'].ffill()  # 직전 값으로 채우기
    oil.index.name = "date"

    # 3) 이상치 탐지 (Z-score)
    window = 7 
    rolling_mean = oil['brent_usd'].rolling(window=window, center=True).mean()
    rolling_std  = oil['brent_usd'].rolling(window=window, center=True).std()

    oil['z_score']   = (oil['brent_usd'] - rolling_mean) / rolling_std
    oil['z_outlier'] = oil['z_score'].abs() > 3

    print("=== Z-Score Outlier Detection ===")
    print("Total days:", len(oil))
    print("Z-Score outliers flagged:", int(oil['z_outlier'].sum()), "rows\n") # 이상치 없음
    print(oil[oil['z_outlier']].head(), "\n")

    # 4) 등락률 계산 & 이벤트 플래그
    oil["brent_prev7"] = oil["brent_usd"].shift(7)
    oil["pct_chg_7d"]  = (oil["brent_usd"] - oil["brent_prev7"]) / oil["brent_prev7"]

    oil["oil_spike"] = oil["pct_chg_7d"].abs() >= 0.05 # 5% 이상 등락 → 이벤트
    oil["oil_rise"]  = oil["pct_chg_7d"] >=  0.05
    oil["oil_fall"]  = oil["pct_chg_7d"] <= -0.05

    # 5) 요약 & 저장 
    print("After processing:")
    print("Total days:", len(oil))
    print("Spike events:", int(oil["oil_spike"].sum()), "\n")

    out_path = data_dir/"oil_price_processed.csv"
    oil.reset_index().to_csv(
        out_path, index=False, date_format="%Y-%m-%d"
    )
    print(f"Saved processed data to {out_path}")

if __name__ == "__main__":
    main()