# src/models/simple_baseline_model.py
# 베이스라인과 유사한 단순한 모델

import pandas as pd
import numpy as np
import sqlite3
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# 경로 설정
SCRIPT = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT.parents[2]
DATA_DIR = PROJECT_ROOT / "data"

def get_country_mapping():
    """국가 매핑"""
    return {
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

def load_simple_data():
    """단순한 데이터 로드"""
    # 수요 데이터
    conn = sqlite3.connect(DATA_DIR / "demand_train_processed.db")
    demand = pd.read_sql("SELECT * FROM demand_train", conn, parse_dates=['date'])
    conn.close()
    
    # 국가 매핑 추가
    country_map = get_country_mapping()
    demand["country"] = demand["city"].map(country_map)
    
    # 기본 피처만
    demand["year"] = demand["date"].dt.year
    demand["month"] = demand["date"].dt.month
    demand["dayofyear"] = demand["date"].dt.dayofyear
    demand["weekday"] = demand["date"].dt.weekday
    
    # SKU 메타 정보
    sku_meta = pd.read_csv(DATA_DIR / "sku_meta.csv", parse_dates=["launch_date"])
    demand = demand.merge(sku_meta[["sku", "family", "storage_gb", "launch_date"]], on="sku", how="left")
    demand["days_since_launch"] = (demand["date"] - demand["launch_date"]).dt.days.clip(lower=0)
    
    # 간단한 랙 피처만
    demand = demand.sort_values(["city", "sku", "date"])
    demand["demand_lag_7"] = demand.groupby(["city", "sku"])["demand"].shift(7).fillna(0)
    demand["demand_lag_30"] = demand.groupby(["city", "sku"])["demand"].shift(30).fillna(0)
    
    return demand

def train_simple_model():
    """단순한 모델 학습"""
    print("Loading data...")
    data = load_simple_data()
    
    # 2018-2022 전체 데이터 사용 (베이스라인과 동일)
    train_mask = (data["date"] >= "2018-01-01") & (data["date"] <= "2022-12-31")
    train_data = data[train_mask].copy()
    
    print(f"Training data: {len(train_data):,} samples")
    
    # 단순한 피처만 사용
    feature_cols = [
        "city", "sku", "country", "family", 
        "month", "dayofyear", "weekday", "storage_gb",
        "days_since_launch", "demand_lag_7", "demand_lag_30"
    ]
    
    # 범주형 인코딩
    label_encoders = {}
    for col in ["city", "sku", "country", "family"]:
        le = LabelEncoder()
        train_data[col + "_encoded"] = le.fit_transform(train_data[col].astype(str))
        label_encoders[col] = le
        feature_cols.append(col + "_encoded")
        feature_cols.remove(col)
    
    # 모델 학습 (단순한 Random Forest)
    X_train = train_data[feature_cols]
    y_train = train_data["demand"]
    
    model = RandomForestRegressor(
        n_estimators=50,  # 트리 수 줄임
        max_depth=10,     # 깊이 줄임
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    print("Model trained successfully!")
    return model, label_encoders, feature_cols

def generate_simple_forecast():
    """단순한 예측 생성"""
    print("Training simple model...")
    model, label_encoders, feature_cols = train_simple_model()
    
    print("Loading forecast template...")
    template = pd.read_csv(DATA_DIR / "forecast_submission_template.csv", parse_dates=["date"])
    
    # 기본 피처 추가
    country_map = get_country_mapping()
    template["country"] = template["city"].map(country_map)
    template["year"] = template["date"].dt.year
    template["month"] = template["date"].dt.month
    template["dayofyear"] = template["date"].dt.dayofyear
    template["weekday"] = template["date"].dt.weekday
    
    # SKU 메타 정보
    sku_meta = pd.read_csv(DATA_DIR / "sku_meta.csv", parse_dates=["launch_date"])
    template = template.merge(sku_meta[["sku", "family", "storage_gb", "launch_date"]], on="sku", how="left")
    template["days_since_launch"] = (template["date"] - template["launch_date"]).dt.days.clip(lower=0)
    
    # 랙 피처 (2022년 평균값으로 대체)
    conn = sqlite3.connect(DATA_DIR / "demand_train_processed.db")
    recent_data = pd.read_sql("SELECT * FROM demand_train WHERE date >= '2022-01-01'", conn, parse_dates=['date'])
    conn.close()
    
    recent_avg = recent_data.groupby(["city", "sku"])["demand"].mean()
    template = template.set_index(["city", "sku"]).join(recent_avg.rename("recent_avg")).reset_index()
    template["recent_avg"] = template["recent_avg"].fillna(recent_data["demand"].mean())
    
    # 간단한 랙 피처들
    template["demand_lag_7"] = template["recent_avg"] * 0.9
    template["demand_lag_30"] = template["recent_avg"] * 0.8
    
    # 범주형 인코딩
    for col, le in label_encoders.items():
        try:
            template[col + "_encoded"] = le.transform(template[col].astype(str))
        except ValueError:
            template[col + "_encoded"] = 0
    
    # 예측
    print("Generating predictions...")
    X_pred = template[feature_cols]
    predictions = model.predict(X_pred)
    
    # 결과 정리
    template["mean"] = np.maximum(0, predictions.round().astype(int))
    result = template[["date", "sku", "city", "mean"]].copy()
    
    return result

def main():
    """메인 실행"""
    print("=== Simple Baseline Model ===\n")
    
    # 예측 생성
    forecast = generate_simple_forecast()
    
    # 결과 저장
    output_path = DATA_DIR / "simple_baseline_forecast.csv"
    forecast.to_csv(output_path, index=False)
    print(f"\nForecast saved to: {output_path}")
    
    # 결과 요약
    print(f"\n=== Forecast Summary ===")
    print(f"Total predictions: {len(forecast):,}")
    print(f"Date range: {forecast['date'].min()} to {forecast['date'].max()}")
    print(f"Average daily demand: {forecast['mean'].mean():.1f}")
    print(f"Max daily demand: {forecast['mean'].max():,}")
    print(f"Min daily demand: {forecast['mean'].min()}")
    
    return forecast

if __name__ == "__main__":
    forecast = main() 