# src/data/check_missing_outliers.py

import pandas as pd
from pathlib import Path

def report(df: pd.DataFrame, name: str):
    print(f"\n=== {name} ===")
    # 1) 결측치
    missing = df.isna().sum()
    print("Missing values per column:")
    print(missing[missing > 0])

    # 2) 수치형 컬럼 이상치 (IQR 기준)
    num = df.select_dtypes(include="number")
    if not num.empty:
        print("\nOutliers (IQR 1.5x) per numeric column:")
        Q1 = num.quantile(0.25)
        Q3 = num.quantile(0.75)
        IQR = Q3 - Q1
        outlier_counts = ((num < (Q1 - 1.5 * IQR)) | (num > (Q3 + 1.5 * IQR))).sum()
        print(outlier_counts[outlier_counts > 0])

    # 3) 범주형 컬럼 유니크값
    cat = df.select_dtypes(include="object")
    if not cat.empty:
        print("\nUnique values per categorical column:")
        for col in cat.columns:
            print(f"  {col}: {df[col].nunique()} unique")

def main():
    data_dir = Path(__file__).parents[2] / "data"

    files = {
        "calendar":           ("calendar.csv",    {"parse_dates":["date"]}),
        "holiday_lookup":     ("holiday_lookup.csv", {"parse_dates":["date"]}),
        "weather":            ("weather.csv",     {"parse_dates":["date"]}),
        "oil_price":          ("oil_price.csv",   {"parse_dates":["date"]}),
        "currency":           ("currency.csv",    {"parse_dates":["Date"]}),
        "consumer_confidence":("consumer_confidence.csv", {"parse_dates":["month"]}),
        "sku_meta":           ("sku_meta.csv",    {"parse_dates":["launch_date"]}),
        "price_promo_train":  ("price_promo_train.csv", {"parse_dates":["date"]}),
        "marketing_spend":    ("marketing_spend.csv", {"parse_dates":["date"]}),
    }

    for name, (fname, kwargs) in files.items():
        path = data_dir / fname
        df = pd.read_csv(path, **kwargs)
        # currency.csv Date 컬럼 이름 통일
        if name == "currency":
            df = df.rename(columns={"Date":"date"})
        report(df, name)

if __name__ == "__main__":
    main()
