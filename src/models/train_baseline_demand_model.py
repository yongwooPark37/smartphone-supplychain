# src/models/train_baseline_demand_model.py

import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error

# === 경로 견고화 ===
SCRIPT = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT.parents[2]
DATA_DIR = PROJECT_ROOT / "data"
# ======================


def load_features() -> pd.DataFrame:
    """
    DATA_DIR로부터 전처리된 피처(유가, 환율, 소비심리지수,
    SKU 메타, 프로모션, 마케팅 지출, 계절)를 로드하고,
    demand_train 테이블과 합쳐 학습용 DataFrame을 반환합니다.
    """
    # -- 데이터 로드 --
    oil = pd.read_csv(DATA_DIR / "oil_price_processed.csv", parse_dates=["date"])
    curr = pd.read_csv(DATA_DIR / "currency_processed.csv", parse_dates=["date"])
    cc = pd.read_csv(DATA_DIR / "consumer_confidence_processed.csv", parse_dates=["date"])
    sku = pd.read_csv(DATA_DIR / "sku_meta.csv", parse_dates=["launch_date"])
    promo = pd.read_csv(DATA_DIR / "price_promo_train.csv", parse_dates=["date"])
    ms = pd.read_csv(DATA_DIR / "marketing_spend.csv", parse_dates=["date"])
    cal = pd.read_csv(DATA_DIR / "calendar.csv", parse_dates=["date"])

    # demand_train 로드
    conn = sqlite3.connect(DATA_DIR / "demand_train.db")
    dem = pd.read_sql(
        "SELECT date, city, sku, demand FROM demand_train", 
        conn, parse_dates=["date"]
    )
    conn.close()

    # 도시→국가 매핑
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
    dem['country'] = dem['city'].map(country_map)

    # 기본 DataFrame
    df = dem.copy()
    # 1) 유가 병합
    df = df.merge(oil[['date','brent_usd','oil_spike']], on='date', how='left')

    # 2) 환율 멀티 통화 지원
    curr_long = curr.melt(id_vars='date', var_name='currency', value_name='rate')
    currency_code_map = {
        'USA':'USD','DEU':'EUR','FRA':'EUR','KOR':'KRW','JPN':'JPY',
        'GBR':'GBP','CAN':'CAD','AUS':'AUD','BRA':'BRL','ZAF':'ZAR'
    }
    df['currency'] = df['country'].map(currency_code_map)
    df = df.merge(
        curr_long,
        left_on=['date','currency'],
        right_on=['date','currency'],
        how='left'
    ).rename(columns={'rate':'fx_rate'})

    # 3) 소비심리지수 병합
    df = df.merge(cc[['date','country','confidence_index']], on=['date','country'], how='left')
    # 4) SKU 메타
    df = df.merge(
        sku[['sku','family','storage_gb','life_days','launch_date']],
        on='sku', how='left'
    )
    # 5) 프로모션
    df = df.merge(
        promo[['date','sku','city','unit_price']],
        on=['date','sku','city'], how='left'
    )
    # 6) 마케팅 지출
    df = df.merge(
        ms[['date','country','spend_usd']],
        on=['date','country'], how='left'
    )
    # 7) 계절(merge 대신 map)
    season_map = dict(zip(cal['date'], cal['season']))
    df['season'] = df['date'].map(season_map)

    # 결측 보간
    df['unit_price'] = df['unit_price'].ffill()
    df['spend_usd'] = df['spend_usd'].fillna(0)
    df['fx_rate'] = df['fx_rate'].ffill()

    # 경과 일수
    df['days_since_launch'] = (
        df['date'] - df['launch_date']
    ).dt.days.clip(lower=0)

    return df


def train_baseline_with_scaling(df: pd.DataFrame):
    # 2018-2022 데이터 필터링
    mask = (df['date'] >= '2018-01-01') & (df['date'] <= '2022-12-31')
    data = df.loc[mask].copy()

    # FX결측 보간
    data['fx_rate'] = data['fx_rate'].fillna(method='ffill').fillna(method='bfill')

    # 피처/타겟 분리
    X = data.drop(columns=['demand','date','launch_date','currency'])  # currency는 이미 fx_rate로 대체
    y = data['demand']

    # fx_rate 스케일링
    scaler = StandardScaler()
    X[['fx_rate']] = scaler.fit_transform(X[['fx_rate']])

    # 범주형 인코딩
    for col in ['city','country','sku','family','season']:
        X[col] = LabelEncoder().fit_transform(X[col].astype(str))

    # 학습/검증 분리
    train_idx = data['date'] < '2022-01-01'
    X_train, y_train = X.loc[train_idx], y.loc[train_idx]
    X_val, y_val   = X.loc[~train_idx], y.loc[~train_idx]

    # 모델 학습
    model = HistGradientBoostingRegressor(max_iter=200)
    model.fit(X_train, y_train)

    # 검증
    preds = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, preds))
    print(f"Baseline with fx_rate scaling RMSE: {rmse:.2f}")
    return model, scaler


if __name__ == '__main__':
    df = load_features()
    train_baseline_with_scaling(df)
