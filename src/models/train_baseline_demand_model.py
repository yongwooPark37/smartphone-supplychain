# src/models/train_baseline_demand_model.py

from typing import Dict, List, Tuple
import sqlite3
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from xgboost import XGBRegressor

# 경로 견고화
import sys
from pathlib import Path
SCRIPT = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT.parents[2]
SRC_DIR = PROJECT_ROOT / "src"
DATA_DIR = PROJECT_ROOT / "data"
sys.path.append(str(SRC_DIR))

from data.feature_engineering import add_time_series_features
from data.process_demand import detect_soft_event_score

def load_features() -> pd.DataFrame:
    # 1) 공통 피처 로드
    oil   = pd.read_csv(DATA_DIR/"oil_price_processed.csv", parse_dates=["date"])
    curr  = pd.read_csv(DATA_DIR/"currency_processed.csv",    parse_dates=["date"])
    cc    = pd.read_csv(DATA_DIR/"consumer_confidence_processed.csv", parse_dates=["date"])
    sku   = pd.read_csv(DATA_DIR/"sku_meta.csv",              parse_dates=["launch_date"])
    promo = pd.read_csv(DATA_DIR/"price_promo_train.csv",     parse_dates=["date"])
    ms    = pd.read_csv(DATA_DIR/"marketing_spend.csv",       parse_dates=["date"])
    cal   = pd.read_csv(DATA_DIR/"calendar.csv",              parse_dates=["date"])

    # 2) demand_train_processed 로드
    db_path = DATA_DIR/"demand_train_processed.db"
    conn = sqlite3.connect(db_path)
    dem  = pd.read_sql(
        "SELECT date, city, sku, demand FROM demand_train",
        conn, parse_dates=["date"]
    )
    conn.close()

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

    # 3) city→country 매핑
    dem["country"] = dem["city"].map(country_map)

    # 4) 기본 병합
    df = dem.copy()
    df = df.merge(oil[["date","brent_usd","oil_spike"]], on="date", how="left")

    curr_long = (
        curr.melt(id_vars="date", var_name="raw_currency", value_name="rate")
            .assign(currency=lambda d: d["raw_currency"].str.replace("=X$", "", regex=True))
    )
    code_map = {
        'USA':'USD','DEU':'EUR','FRA':'EUR','KOR':'KRW','JPN':'JPY',
        'GBR':'GBP','CAN':'CAD','AUS':'AUD','BRA':'BRL','ZAF':'ZAR'
    }
    df["currency"] = df["country"].map(code_map)
    df = df.merge(
        curr_long[["date","currency","rate"]],
        on=["date","currency"], how="left"
    ).rename(columns={"rate":"fx_rate"})
    df.loc[df["currency"]=="USD", "fx_rate"] = 1.0

    df = df.merge(cc[["date","country","confidence_index"]], on=["date","country"], how="left")
    df = df.merge(sku[["sku","family","storage_gb", "launch_date"]], on="sku", how="left")
    df = df.merge(promo[["date","sku","city","unit_price"]],         on=["date","sku","city"], how="left")
    df = df.merge(ms[["date","country","spend_usd"]],               on=["date","country"],   how="left")
    df = df.merge(cal[["date","country","season"]],                 on=["date","country"],   how="left")

    # 5) 결측 보간
    df["unit_price"] = df["unit_price"].ffill()
    df["spend_usd"] = df["spend_usd"].fillna(0)
    df["fx_rate"] = df["fx_rate"].ffill()

    # 6) 할인 프로모션 플래그
    promo = pd.read_csv(DATA_DIR/"price_promo_train.csv", parse_dates=["date"])
    promo["promo_flag"] = (promo["discount_pct"] > 0).astype(int)
    df = df.merge(promo[["date","sku","city","promo_flag"]], on=["date","sku","city"], how="left")
    df["promo_flag"] = df["promo_flag"].fillna(0)

    # 7) 시계열 랙·롤링 피처
    df = add_time_series_features(df, lags=[1,7,14], rolls=[7,14])

    # 8) days_since_launch
    df["days_since_launch"] = (df["date"] - df["launch_date"]).dt.days.clip(lower=0)

    # 9) 신제품 공개 행사
    soft_df = detect_soft_event_score(
        df=dem,
        sku_meta_path=DATA_DIR/"sku_meta.csv",
        country_map=country_map,
        threshold_pct=0.20,
        min_days=10,
        max_days=90
    )
    df = df.merge(soft_df, on=["date","country"], how="left")

    return df

def train_baseline_with_xgb(df: pd.DataFrame):
    """2018–2021 학습, 2022 검증 → RMSE/R2 출력 (XGBoost 버전)"""
    mask = (df["date"] >= "2018-01-01") & (df["date"] <= "2022-12-31")
    data = df.loc[mask].dropna().copy()

    # 피처/타겟 분리
    X = data.drop(columns=["demand","date","launch_date","currency"])
    X = X.drop(columns=["demand_roll_14"])
    X = X.drop(columns=["confidence_index"])
    print(X.head(5))
    y = data["demand"]

    # 수치형 스케일링
    num_cols = [
        "brent_usd","unit_price",
        "demand_lag_1","demand_lag_7","demand_lag_14",
        "demand_roll_7"
    ]
    X[num_cols] = StandardScaler().fit_transform(X[num_cols])

    # 범주형 인코딩
    for col in ["city","country","sku","family","season","promo_flag"]:
        X[col] = LabelEncoder().fit_transform(X[col].astype(str))
    
    print(X.head(5))

    # 학습/검증 분할
    split = "2022-01-01"
    train_idx = data["date"] < split
    X_train, y_train = X.loc[train_idx], y.loc[train_idx]
    X_val,   y_val   = X.loc[~train_idx], y.loc[~train_idx]

    # XGBoost 모델에 eval_metric 을 생성자에 넘기기
    model = XGBRegressor(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        tree_method="hist"
    )
    model.fit(X_train, y_train)

    # 예측 & 지표 계산
    preds = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, preds))
    r2   = r2_score(y_val, preds)
    print(f"XGBoost Validation RMSE: {rmse:.2f}")
    print(f"XGBoost Validation R²:   {r2:.3f}")

    # 검증용 데이터에 예측값 붙여서 리턴
    val = data.loc[~train_idx].copy()
    val["pred"] = preds
    return model, val

if __name__ == "__main__":
    # (1) 특성 로드 + 모델 학습 + 검증셋, 예측값 얻기
    df = load_features()
    model, val = train_baseline_with_xgb(df)

    # (2) 날짜별 TOTAL 수요 합계 집계
    agg = (
        val.groupby("date")
           .agg(actual   = ("demand","sum"),
                predicted= ("pred","sum"))
           .reset_index()
    )

    # (3) 플롯
    plt.figure(figsize=(12,4))
    plt.plot(agg["date"], agg["actual"],    label="Actual",    linewidth=2)
    plt.plot(agg["date"], agg["predicted"], label="Predicted", linewidth=2)
    plt.title("2022 Validation: Total Daily Demand (Actual vs Predicted)")
    plt.xlabel("Date"); plt.ylabel("Units Sold")
    plt.legend(); plt.tight_layout()
    plt.show()

