# src/models/fill_forecast_submission.py

import sqlite3
from pathlib import Path
import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBRegressor

# 경로
SCRIPT = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT.parents[2]
SRC_DIR = PROJECT_ROOT / "src"
DATA_DIR = PROJECT_ROOT / "data"

# 학습 함수 불러오기
import sys
sys.path.append(str(SRC_DIR))
from models.train_baseline_demand_model import load_features, train_baseline_with_xgb
from data.feature_engineering import add_time_series_features
from data.process_demand import detect_launch_events

def load_historical_and_train():
    # 1) 학습용 특성 + 모델 학습
    df = load_features()  # 2018–2022까지 전처리된 DataFrame
    model, _ = train_baseline_with_xgb(df)
    return model

def build_forecast_features(template: pd.DataFrame) -> pd.DataFrame:
    # --- (1) 모든 원본 테이블 로드 ---
    oil   = pd.read_csv(DATA_DIR/"oil_price_processed.csv", parse_dates=["date"])
    curr  = pd.read_csv(DATA_DIR/"currency_processed.csv", parse_dates=["date"])
    cc    = pd.read_csv(DATA_DIR/"consumer_confidence_processed.csv", parse_dates=["date"])
    sku   = pd.read_csv(DATA_DIR/"sku_meta.csv", parse_dates=["launch_date"])
    promo = pd.read_csv(DATA_DIR/"price_promo_train.csv", parse_dates=["date"])
    ms    = pd.read_csv(DATA_DIR/"marketing_spend.csv", parse_dates=["date"])
    cal   = pd.read_csv(DATA_DIR/"calendar.csv", parse_dates=["date"])

    # --- (2) 과거 수요 불러오기 + 템플릿 결합 ---
    conn = sqlite3.connect(DATA_DIR/"demand_train_processed.db")
    hist = pd.read_sql("SELECT date, city, sku, demand FROM demand_train",
                       conn, parse_dates=["date"])
    conn.close()

    tmp = template.copy()
    tmp["demand"] = np.nan
    df = pd.concat([hist, tmp], ignore_index=True)
    df.sort_values(["city","sku","date"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    # --- (3) oil price merge ---
    df = df.merge(oil[["date","brent_usd","oil_spike"]],
                  on="date", how="left")

    # --- (4) fx_rate merge ---
    curr_long = (
        curr.melt(id_vars="date", var_name="raw_currency", value_name="rate")
            .assign(currency=lambda d: d["raw_currency"].str.replace("=X$", "", regex=True))
    )
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
    df["country"]  = df["city"].map(country_map)
    code_map = {
        'USA':'USD','DEU':'EUR','FRA':'EUR','KOR':'KRW','JPN':'JPY',
        'GBR':'GBP','CAN':'CAD','AUS':'AUD','BRA':'BRL','ZAF':'ZAR'
    }
    df["currency"] = df["country"].map(code_map)
    df = df.merge(curr_long[["date","currency","rate"]],
                  on=["date","currency"], how="left") \
           .rename(columns={"rate":"fx_rate"})
    df.loc[df["currency"]=="USD", "fx_rate"] = 1.0

    # --- (5) consumer confidence merge ---
    df = df.merge(cc[["date","country","confidence_index"]],
                  on=["date","country"], how="left")

    # --- (6) SKU meta merge ---
    df = df.merge(sku[["sku","family","storage_gb","life_days","launch_date"]],
                  on="sku", how="left")

    # --- (7) promo unit_price & promo_flag merge ---
    df = df.merge(promo[["date","sku","city","unit_price","discount_pct"]],
                  on=["date","sku","city"], how="left")
    df["unit_price"].ffill(inplace=True)
    df["promo_flag"] = (df["discount_pct"] > 0).astype(int)
    df.drop(columns=["discount_pct"], inplace=True)

    # --- (8) marketing spend merge ---
    df = df.merge(ms[["date","country","spend_usd"]],
                  on=["date","country"], how="left")
    df["spend_usd"].fillna(0, inplace=True)

    # --- (9) season merge ---
    df = df.merge(cal[["date","country","season"]],
                  on=["date","country"], how="left")

    # --- (10) 시계열 랙·롤링 피처 추가 ---
    df = add_time_series_features(df, lags=[1,7,14], rolls=[7,14])

    # --- (11) days_since_launch ---
    df["days_since_launch"] = (df["date"] - df["launch_date"])\
                                .dt.days.clip(lower=0)

    # --- (12) launch_event 플래그 추가 ---
    events = detect_launch_events(
        df[["date","city","sku","demand"]],  # 내부에서 launch_date 합침
        sku_meta_path=DATA_DIR/"sku_meta.csv",
        country_map=country_map,
        threshold=1.5,
        window=15
    )
    df["launch_event"] = 0
    for country, periods in events.items():
        for s,e in periods:
            mask = (df["country"]==country)&(df["date"]>=s)&(df["date"]<=e)
            df.loc[mask, "launch_event"] = 1

    return df

def main():
    # (1) 모델 로드
    model = load_historical_and_train()

    # (2) 예측 템플릿 불러오기
    tmpl = pd.read_csv(DATA_DIR/"forecast_submission_template.csv",
                       parse_dates=["date"])

    # (3) 피처 빌드
    Xf = build_forecast_features(tmpl)

    # --- (4) 스케일링 및 인코딩 ---
    num_cols = [
        "brent_usd","unit_price","spend_usd","confidence_index",
        "demand_lag_1","demand_lag_7","demand_lag_14",
        "demand_roll_7","demand_roll_14",
        "days_since_launch"
    ]
    # 실제 존재하는 칼럼만 스케일
    exist_num = [c for c in num_cols if c in Xf.columns]
    Xf[exist_num] = StandardScaler().fit_transform(Xf[exist_num])

    for col in ["city","country","sku","family","season","promo_flag","launch_event"]:
        Xf[col] = LabelEncoder().fit_transform(Xf[col].astype(str))

    mask_forecast = Xf["demand"].isna()
    drop_cols = ["date","launch_date","currency","demand","mean"]
    Xf = Xf.drop(columns=drop_cols, errors="ignore")

    X_pred = Xf.loc[mask_forecast].copy()
    X_pred = X_pred[model.get_booster().feature_names]

    # --- (5) 예측 + 정수 변환 ---
    preds = model.predict(X_pred)
    tmpl["demand"] = np.round(preds).astype(int)

    # --- (6) 저장 ---
    out = PROJECT_ROOT/"forecast_submission_filled.csv"
    tmpl.to_csv(out, index=False)
    print(f"Saved filled submission to {out}")

if __name__=="__main__":
    main()
