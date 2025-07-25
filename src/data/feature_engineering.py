# src/data/feature_engineering.py

import pandas as pd
import numpy as np
import sqlite3
from pathlib import Path

def add_event_and_promo_flags(df: pd.DataFrame, data_dir: Path, window: int = 15) -> pd.DataFrame:
    """
    - 프로모션 플래그: discount_pct > 0 → promo_flag=1
    - 신제품 출시 이벤트 플래그: 
        1) 월별 국가 수요 z-score 상위 5%인 월을 후보로
        2) ±window일 범위에 launch_event=1
    """
    # 1) promo_flag
    promo = pd.read_csv(data_dir/"price_promo_train.csv", parse_dates=["date"])
    promo["promo_flag"] = (promo["discount_pct"] > 0).astype(int)
    promo = promo[["date","sku","city","promo_flag"]]
    df = df.merge(promo, on=["date","sku","city"], how="left")
    df["promo_flag"] = df["promo_flag"].fillna(0)
    
    # 2) launch_event
    conn = sqlite3.connect(data_dir/"demand_train.db")
    dem = pd.read_sql("SELECT date, city, sku, demand FROM demand_train", conn, parse_dates=["date"])
    conn.close()
    # city → country 매핑 (load_features와 동일)
    country_map = {
        'Washington_DC':'USA','New_York':'USA','Chicago':'USA','Dallas':'USA',
        'Berlin':'DEU','Munich':'DEU','Frankfurt':'DEU','Hamburg':'DEU',
        'Paris':'FRA','Lyon':'FRA','Marseille':'FRA','Toulouse':'FRA',
        'Seoul':'KOR','Busan':'KOR','Incheon':'KOR','Gwangju':'KOR',
        'Tokyo':'JPN','Osaka':'JPN','Nagoya':'JPN','Fukuoka':'JPN',
        'Manchester':'GBR','London':'GBR','Birmingham':'GBR','Glasgow':'GBR',
        'Ottawa':'CAN','Toronto':'CAN','Vancouver':'CAN','Montreal':'CAN',
        'Canberra':'AUS','Sydney':'AUS','Melbourne':'AUS','Brisbane':'AUS',
        'Brasilia':'BRA','Sao_Paulo':'BRA','Rio_de_Janeiro':'BRA','Salvador':'BRA',
        'Pretoria':'ZAF','Johannesburg':'ZAF','Cape_Town':'ZAF','Durban':'ZAF'
    }
    dem["country"] = dem["city"].map(country_map)
    # 월별 국가 수요 집계
    dem["month"] = dem["date"].dt.to_period("M")
    monthly = (
        dem.groupby(["country","month"])["demand"]
           .sum()
           .reset_index()
    )
    # z-score 계산
    monthly["z_m"] = (
        (monthly["demand"] - monthly.groupby("country")["demand"].transform("mean"))
        / monthly.groupby("country")["demand"].transform("std")
    )
    # 상위 5% 월 추출
    thresh = monthly["z_m"].quantile(0.95)
    events = monthly[monthly["z_m"] >= thresh][["country","month"]]

    # 날짜 단위 확장
    event_days = []
    for _, row in events.iterrows():
        start = row["month"].to_timestamp() - pd.Timedelta(days=window)
        end   = row["month"].to_timestamp("M") + pd.Timedelta(days=window)
        dates = pd.date_range(start, end, freq="D")
        event_days.append(
            pd.DataFrame({
                "date":       dates,
                "country":    row["country"],
                "launch_event": 1
            })
        )
    if event_days:
        evdf = pd.concat(event_days, ignore_index=True)
        df = df.merge(evdf, on=["date","country"], how="left")
        df["launch_event"] = df["launch_event"].fillna(0)
    else:
        df["launch_event"] = 0

    return df


def add_time_series_features(df: pd.DataFrame, lags=[1,7,14], rolls=[7,14]) -> pd.DataFrame:
    """
    - demand 기준 자기회귀(lag) 피처
    - rolling 평균 피처
    """
    df = df.sort_values(["city","sku","date"]).copy()
    for lag in lags:
        df[f"demand_lag_{lag}"] = df.groupby(["city","sku"])["demand"]\
                                      .shift(lag)
    for window in rolls:
        df[f"demand_roll_{window}"] = (
            df.groupby(["city","sku"])["demand"]
              .shift(1)
              .rolling(window, min_periods=1)
              .mean()
        )
    # 결측치는 0 또는 앞뒤 채우기
    df.fillna({f"demand_lag_{lag}":0 for lag in lags}, inplace=True)
    df.fillna({f"demand_roll_{w}":0 for w in rolls}, inplace=True)
    return df
