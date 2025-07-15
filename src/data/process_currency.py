import pandas as pd
from pathlib import Path

def main():
    """
    1) currency.csv 로드
    2) 결측치 확인 및 주말·공휴일 결측 보간
    3) 이상치 탐지 (7-days window 기준 |Z-Score| > 3이면 이상치로 간주)
    4) 이상치 개수·샘플 출력
    5. 결과 저장 (currency_processed.csv)
    """
    
    # 1) 데이터 로드 & 인덱스 설정
    data_dir = Path(__file__).parents[2] / "data"
    curr = (
        pd.read_csv(data_dir/"currency.csv", parse_dates=["Date"])
          .rename(columns={"Date":"date"})
          .set_index("date")
          .sort_index()
    )

    # 2) 결측치 확인 및 보간
    print("=== Missing Before Fill ===")
    print(curr.isna().sum(), "\n")

    full_idx = pd.date_range(curr.index.min(), curr.index.max(), freq="D")
    curr = curr.reindex(full_idx).ffill()
    curr.index.name = "date"

    print("=== Missing After Fill ===")
    print(curr.isna().sum(), "\n")

    # 3) 이상치 탐지
    window = 30
    roll_mean = curr.rolling(window, center=True).mean()
    roll_std  = curr.rolling(window, center=True).std()

    z_scores = (curr - roll_mean) / roll_std
    z_outlier = z_scores.abs() > 3

    # 4) 요약 출력
    print("=== Z-Score Outlier Counts ===")
    print(z_outlier.sum(), "\n")  # 컬럼별 이상치 개수

    for col in curr.columns:
        print(f"--- {col} outlier sample ---")
        print(curr[z_outlier[col]][col].head(), "\n")

    # 5) 결과 저장
    out = curr.copy()
    for col in curr.columns:
        out[f"{col.replace('=X','')}_z_outlier"] = z_outlier[col].astype(int)

    out_path = data_dir/"currency_processed.csv"
    out.reset_index().to_csv(
        out_path, index=False, date_format="%Y-%m-%d"
    )
    print(f"Processed currency saved to {out_path}")

if __name__ == "__main__":
    main()