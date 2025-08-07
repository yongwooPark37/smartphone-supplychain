import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

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

    # 캐나다가 컬럼에 없더라도 추가하지 않고 NaN은 그대로 유지
    if "CAN" not in cc["country"].unique():
        print("⚠️ 캐나다(CAN) 소비자신뢰지수 없음 → NaN 상태로 유지됩니다.\n")

    # 시각화: 월별 국가별 소비자신뢰지수
    plt.figure(figsize=(12, 6))
    for country in cc['country'].unique():
        subset = cc[cc['country'] == country]
        plt.plot(subset['month'], subset['confidence_index'], label=country)

    plt.title("Monthly Consumer Confidence Index by Country")
    plt.xlabel("Month")
    plt.ylabel("Confidence Index")
    plt.legend(title="Country")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # 2) Z-Score 기반 이상치 검출
    # NaN은 자동으로 제외되어 Z-score 계산되지 않음
    cc["mean_cc"] = cc.groupby("country")["confidence_index"].transform("mean")
    cc["std_cc"]  = cc.groupby("country")["confidence_index"].transform("std")
    cc["z_score"] = (cc["confidence_index"] - cc["mean_cc"]) / cc["std_cc"]
    cc["outlier"] = cc["z_score"].abs() > 3

    print("=== Outlier Counts by Country ===")
    print(cc.groupby("country")["outlier"].sum(), "\n")

    # 3) 월간 데이터를 일별로 확장 (NaN도 그대로 복제됨)
    daily_rows = []
    for _, row in cc.iterrows():
        start = row["month"].replace(day=1)
        end = start + pd.offsets.MonthEnd(0)
        dates = pd.date_range(start, end, freq="D")
        df = pd.DataFrame({
            "date": dates,
            "country": row["country"],
            "confidence_index": row["confidence_index"],  # NaN 허용
            "outlier": row["outlier"]
        })
        daily_rows.append(df)

    daily = pd.concat(daily_rows, ignore_index=True)

    # 결과 저장
    out_path = data_dir/"consumer_confidence_processed.csv"
    daily.drop(columns=["outlier"]).to_csv(out_path, index=False, date_format="%Y-%m-%d")
    print(f"Saved daily consumer confidence data to {out_path}, total rows: {len(daily)}")

if __name__ == "__main__":
    main()
