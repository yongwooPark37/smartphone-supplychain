# src/models/simplified_final_model.py
# 단순화된 최종 모델 - 핵심 피처만 사용

import pandas as pd
import numpy as np
import sqlite3
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
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

def get_event_multipliers():
    """실제 과거 데이터 분석 결과를 바탕으로 한 배수"""
    return {
        '2023': {
            'CAN': ('2023-09-01', '2023-11-30', 2.57),
            'DEU': ('2023-11-01', '2023-12-31', 2.30),
            'BRA': ('2023-10-01', '2023-12-31', 2.34),
        },
        '2024': {
            'DEU': ('2024-07-01', '2024-09-30', 2.30),
            'JPN': ('2024-07-01', '2024-09-30', 2.21),
            'GBR': ('2024-07-01', '2024-09-30', 2.39),
        }
    }

def load_training_data():
    """학습 데이터 로드 (핵심 피처만)"""
    # 수요 데이터
    conn = sqlite3.connect(DATA_DIR / "demand_train_processed.db")
    demand = pd.read_sql("SELECT * FROM demand_train", conn, parse_dates=['date'])
    conn.close()
    
    # 국가 매핑 추가
    country_map = get_country_mapping()
    demand["country"] = demand["city"].map(country_map)
    
    # 핵심 시간 피처만
    demand["year"] = demand["date"].dt.year
    demand["month"] = demand["date"].dt.month
    demand["dayofyear"] = demand["date"].dt.dayofyear
    demand["weekday"] = demand["date"].dt.weekday
    demand["quarter"] = demand["date"].dt.quarter
    
    # 계절 정보 추가 (calendar.csv에서)
    calendar = pd.read_csv(DATA_DIR / "calendar.csv", parse_dates=["date"])
    demand = demand.merge(calendar[["date", "country", "season"]], on=["date", "country"], how="left")
    
    # SKU 메타 정보
    sku_meta = pd.read_csv(DATA_DIR / "sku_meta.csv", parse_dates=["launch_date"])
    demand = demand.merge(sku_meta[["sku", "family", "storage_gb", "launch_date"]], on="sku", how="left")
    demand["days_since_launch"] = (demand["date"] - demand["launch_date"]).dt.days.clip(lower=0)
    
    # 핵심 시계열 피처만
    demand = demand.sort_values(["city", "sku", "date"])
    
    # 주요 랙 피처만
    for lag in [7, 30]:
        demand[f"demand_lag_{lag}"] = demand.groupby(["city", "sku"])["demand"].shift(lag).fillna(0)
    
    # 간단한 롤링 평균만
    demand["demand_rolling_mean_7"] = demand.groupby(["city", "sku"])["demand"].transform(lambda x: x.rolling(window=7, min_periods=1).mean())
    
    # 도시별 평균 수요 (기준값)
    city_sku_avg = demand.groupby(["city", "sku"])["demand"].mean().reset_index()
    city_sku_avg = city_sku_avg.rename(columns={"demand": "city_sku_avg_demand"})
    demand = demand.merge(city_sku_avg, on=["city", "sku"], how="left")
    
    return demand

def train_simplified_model(train_data):
    """단순화된 모델 학습"""
    train_mask = train_data["year"] <= 2021
    val_mask = train_data["year"] == 2022
    
    train_set = train_data[train_mask].copy()
    val_set = train_data[val_mask].copy()
    
    # 핵심 피처만 선택
    feature_cols = [
        "city", "sku", "country", "family", 
        "season", "month", "quarter", "dayofyear", "weekday", "storage_gb",
        "days_since_launch", 
        "demand_lag_7", "demand_lag_30",
        "demand_rolling_mean_7",
        "city_sku_avg_demand"
    ]
    
    # 범주형 인코딩
    label_encoders = {}
    for col in ["city", "sku", "country", "family", "season"]:
        le = LabelEncoder()
        train_set[col + "_encoded"] = le.fit_transform(train_set[col].astype(str))
        val_set[col + "_encoded"] = le.transform(val_set[col].astype(str))
        label_encoders[col] = le
        feature_cols.append(col + "_encoded")
        feature_cols.remove(col)
    
    # 모델 학습 - 단순화된 파라미터
    X_train = train_set[feature_cols]
    y_train = train_set["demand"]
    
    model = RandomForestRegressor(
        n_estimators=100,  # 트리 수 줄임
        max_depth=10,      # 깊이 줄임
        min_samples_split=10,  # 분할 기준 강화
        min_samples_leaf=5,    # 리프 노드 최소 샘플 강화
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    # 검증
    X_val = val_set[feature_cols]
    y_val = val_set["demand"]
    val_pred = model.predict(X_val)
    
    from sklearn.metrics import mean_squared_error, r2_score
    rmse = np.sqrt(mean_squared_error(y_val, val_pred))
    r2 = r2_score(y_val, val_pred)
    
    print(f"Simplified Model - Validation RMSE: {rmse:.2f}")
    print(f"Simplified Model - Validation R²: {r2:.3f}")
    
    # 과소평가 문제 진단
    print(f"\n=== Bias Analysis ===")
    print(f"Actual mean: {y_val.mean():.2f}")
    print(f"Predicted mean: {val_pred.mean():.2f}")
    print(f"Bias: {val_pred.mean() - y_val.mean():.2f}")
    print(f"Relative bias: {((val_pred.mean() - y_val.mean()) / y_val.mean() * 100):.1f}%")
    
    # 피처 중요도
    importance_df = pd.DataFrame({
        "feature": X_train.columns,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False)
    
    print("\n=== Top 10 Feature Importance ===")
    print(importance_df.head(10))
    
    # 2022 검증 결과 시각화
    visualize_2022_validation(val_set, val_pred, r2)
    
    return model, label_encoders, feature_cols

def visualize_2022_validation(val_set, val_pred, r2_score):
    """2022 검증 결과 시각화"""
    
    # val_pred를 val_set과 같은 순서로 정렬
    val_set_with_pred = val_set.copy()
    val_set_with_pred["predicted"] = val_pred
    
    # 일별 총 수요 집계
    daily_actual = val_set_with_pred.groupby("date")["demand"].sum().reset_index()
    daily_pred = val_set_with_pred.groupby("date")["predicted"].sum().reset_index()
    
    # 데이터 병합
    comparison = daily_actual.merge(daily_pred, on="date", suffixes=("", "_pred"))
    comparison = comparison.rename(columns={"predicted": "predicted"})
    
    # 시각화 - 하나의 그래프만
    plt.figure(figsize=(12, 6))
    plt.plot(comparison["date"], comparison["demand"], label="Actual", alpha=0.8, linewidth=2, color='blue')
    plt.plot(comparison["date"], comparison["predicted"], label="Predicted", alpha=0.8, linewidth=2, color='red')
    plt.title(f"2022 Daily Demand: Actual vs Predicted (R² = {r2_score:.3f})")
    plt.xlabel("Date")
    plt.ylabel("Daily Total Demand")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def generate_simplified_forecast():
    """단순화된 예측 생성"""
    print("Loading training data...")
    train_data = load_training_data()
    
    print("Training simplified model...")
    model, label_encoders, feature_cols = train_simplified_model(train_data)
    
    print("Generating forecast...")
    
    # 2023-2024 예측 데이터 생성
    future_dates = pd.date_range(start="2023-01-01", end="2024-12-31", freq="D")
    future_df = pd.DataFrame({"date": future_dates})
    
    # 미래 데이터에 피처 추가
    future_df["year"] = future_df["date"].dt.year
    future_df["month"] = future_df["date"].dt.month
    future_df["dayofyear"] = future_df["date"].dt.dayofyear
    future_df["weekday"] = future_df["date"].dt.weekday
    future_df["quarter"] = future_df["date"].dt.quarter
    
    # 계절 정보 추가
    calendar = pd.read_csv(DATA_DIR / "calendar.csv", parse_dates=["date"])
    future_df = future_df.merge(calendar[["date", "country", "season"]], on="date", how="left")
    
    # SKU 정보 추가
    sku_meta = pd.read_csv(DATA_DIR / "sku_meta.csv", parse_dates=["launch_date"])
    
    # 모든 도시와 SKU 조합 생성
    cities = train_data["city"].unique()
    skus = train_data["sku"].unique()
    
    results = []
    events = get_event_multipliers()
    
    for city in cities:
        country = get_country_mapping()[city]
        
        for sku in skus:
            sku_info = sku_meta[sku_meta["sku"] == sku].iloc[0]
            
            # 해당 SKU의 미래 데이터 생성
            sku_future = future_df.copy()
            sku_future["city"] = city
            sku_future["country"] = country
            sku_future["sku"] = sku
            sku_future["family"] = sku_info["family"]
            sku_future["storage_gb"] = sku_info["storage_gb"]
            sku_future["launch_date"] = sku_info["launch_date"]
            sku_future["days_since_launch"] = (sku_future["date"] - sku_future["launch_date"]).dt.days.clip(lower=0)
            
            # 과거 데이터에서 해당 SKU의 최근 수요 정보 가져오기
            sku_history = train_data[(train_data["city"] == city) & (train_data["sku"] == sku)].sort_values("date")
            
            if len(sku_history) > 0:
                # 최근 수요값들로 미래 예측을 위한 피처 생성
                recent_demand = sku_history["demand"].iloc[-1] if len(sku_history) > 0 else 0
                recent_lag_7 = sku_history["demand"].iloc[-7] if len(sku_history) >= 7 else recent_demand
                recent_lag_30 = sku_history["demand"].iloc[-30] if len(sku_history) >= 30 else recent_demand
                recent_rolling_mean = sku_history["demand"].tail(7).mean() if len(sku_history) >= 7 else recent_demand
                city_sku_avg = sku_history["demand"].mean()
            else:
                # 데이터가 없는 경우 기본값
                recent_demand = 0
                recent_lag_7 = 0
                recent_lag_30 = 0
                recent_rolling_mean = 0
                city_sku_avg = 0
            
            # 미래 데이터에 과거 정보 적용
            sku_future["demand"] = recent_demand
            sku_future["demand_lag_7"] = recent_lag_7
            sku_future["demand_lag_30"] = recent_lag_30
            sku_future["demand_rolling_mean_7"] = recent_rolling_mean
            sku_future["city_sku_avg_demand"] = city_sku_avg
            
            # 범주형 인코딩
            for col in ["city", "sku", "country", "family", "season"]:
                le = label_encoders[col]
                sku_future[col + "_encoded"] = le.transform(sku_future[col].astype(str))
            
            # 모델 예측
            X_future = sku_future[feature_cols]
            base_pred = model.predict(X_future)
            
            # 이벤트 배수 적용
            for date, pred in zip(sku_future["date"], base_pred):
                event_multiplier = 1.0
                
                # 이벤트 확인
                for year, year_events in events.items():
                    for event_country, (start_date, end_date, multiplier) in year_events.items():
                        if (event_country == country and 
                            pd.to_datetime(start_date) <= date <= pd.to_datetime(end_date)):
                            event_multiplier = multiplier
                            break
                
                adjusted_pred = pred * event_multiplier
                
                results.append({
                    "date": date.strftime("%Y-%m-%d"),
                    "sku": sku,
                    "city": city,
                    "mean": int(max(adjusted_pred, 0))
                })
    
    # 결과 저장
    result_df = pd.DataFrame(results)
    output_path = DATA_DIR / "simplified_forecast_submission.csv"
    result_df.to_csv(output_path, index=False)
    
    print(f"✅ Simplified forecast saved: {output_path}")
    print(f"Total predictions: {len(result_df):,}")
    print(f"Average demand: {result_df['mean'].mean():.1f}")
    print(f"Max demand: {result_df['mean'].max():,}")
    print(f"Min demand: {result_df['mean'].min()}")
    
    return result_df

def main():
    """메인 실행"""
    print("=== Simplified Final Forecast Model ===\n")
    
    result_df = generate_simplified_forecast()
    
    print(f"\n✅ 단순화된 최종 모델 완료!")
    print(f"📁 결과 파일: simplified_forecast_submission.csv")

if __name__ == "__main__":
    main() 