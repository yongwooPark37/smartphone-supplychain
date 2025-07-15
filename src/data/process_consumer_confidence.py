import pandas as pd
from pathlib import Path

def main():
    """
    1) consumer_confidence.csv 로드 및 결측치 확인
    2) 국가별 Z-Score 기반 이상치 탐지 (|Z| > 3)
    3) 월간 데이터를 일별로 확장 후 저장 (consumer_confidence_processed.csv)
    """

    # 1) 데이터 로드 및 결측치 확인
    data_dir = Path(__file__).parents[2] / "data"
    cc = pd.read_csv(data_dir/"consumer_confidence.csv", parse_dates=["month"])
    print("=== Missing Values ===")
    print(cc.isna().sum(), "\n")

    # 2) Z-Score 기반 이상치 검출
    # country 그룹별 평균과 표준편차 계산
    cc["mean_cc"] = cc.groupby("country")["confidence_index"].transform("mean")
    cc["std_cc"]  = cc.groupby("country")["confidence_index"].transform("std")
    cc["z_score"] = (cc["confidence_index"] - cc["mean_cc"]) / cc["std_cc"]
    # |Z| > 3 이상치를 outlier로 표시
    cc["outlier"] = cc["z_score"].abs() > 3
    print("=== Outlier Counts by Country ===")
    print(cc.groupby("country")["outlier"].sum(), "\n")

    # 3) 월간 데이터를 일별로 확장 후 저장
    daily_rows = []
    for _, row in cc.iterrows():
        start = row["month"].replace(day=1)
        end = start + pd.offsets.MonthEnd(0)
        dates = pd.date_range(start, end, freq="D")
        df = pd.DataFrame({
            "date": dates,
            "country": row["country"],
            "confidence_index": row["confidence_index"],
            "outlier": row["outlier"]
        })
        daily_rows.append(df)
    daily = pd.concat(daily_rows, ignore_index=True)
    # 결과 저장
    out_path = data_dir/"consumer_confidence_processed.csv"
    daily.to_csv(out_path, index=False, date_format="%Y-%m-%d")
    print(f"Saved daily consumer confidence data to {out_path}, total rows: {len(daily)}")


if __name__ == "__main__":
    main()