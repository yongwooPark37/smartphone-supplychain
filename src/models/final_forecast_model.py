# src/models/final_forecast_model.py
# EDA 기반 고급 시계열 예측 모델 - 출제자 접근법 반영

import pandas as pd
import numpy as np
import sqlite3
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.decomposition import PCA
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

def create_global_confidence_factor(consumer_conf):
    """글로벌 신뢰지수 요인 생성 (출제자 방식)"""
    # 피벗 테이블 생성
    wide = consumer_conf.pivot(index="month", columns="country", values="confidence_index").sort_index()
    
    # 결측치 처리
    wide = wide.fillna(method='ffill')
    
    # 표준화
    scaler = StandardScaler()
    Z = scaler.fit_transform(wide)
    
    # PCA로 글로벌 요인 추출
    pca = PCA(n_components=2)
    global_factors = pca.fit_transform(Z)
    
    # 글로벌 요인을 시계열로 변환
    global_factor_df = pd.DataFrame({
        'year_month': wide.index,
        'global_factor_1': global_factors[:, 0],
        'global_factor_2': global_factors[:, 1]
    })
    
    return global_factor_df, pca, scaler

def detect_events_using_zscore(demand, threshold=2.0):
    """Z-score 기반 이벤트 탐지 (출제자 방식)"""
    # 국가별 월별 수요 집계
    demand['year_month'] = demand['date'].dt.to_period('M')
    monthly_country_demand = demand.groupby(['country', 'year_month'])['demand'].sum().reset_index()
    
    events_detected = []
    
    for country in monthly_country_demand['country'].unique():
        country_data = monthly_country_demand[monthly_country_demand['country'] == country].copy()
        country_data = country_data.sort_values('year_month')
        
        # 이동 평균과 표준편차 계산
        country_data['demand_mean'] = country_data['demand'].rolling(window=12, min_periods=1).mean()
        country_data['demand_std'] = country_data['demand'].rolling(window=12, min_periods=1).std()
        country_data['z_score'] = (country_data['demand'] - country_data['demand_mean']) / country_data['demand_std']
        
        # 이벤트 감지
        events = country_data[country_data['z_score'] > threshold]
        
        for _, event in events.iterrows():
            events_detected.append({
                'country': country,
                'date': event['year_month'],
                'demand': event['demand'],
                'z_score': event['z_score'],
                'normal_demand': event['demand_mean'],
                'multiplier': event['demand'] / event['demand_mean']
            })
    
    return pd.DataFrame(events_detected)

def load_enhanced_training_data():
    """EDA 기반 고급 학습 데이터 로드"""
    print("=== EDA 기반 고급 데이터 로드 ===")
    
    # 1. 수요 데이터
    conn = sqlite3.connect(DATA_DIR / "demand_train.db")
    demand = pd.read_sql("SELECT * FROM demand_train", conn, parse_dates=['date'])
    conn.close()
    
    # 국가 매핑 추가
    country_map = get_country_mapping()
    demand["country"] = demand["city"].map(country_map)
    
    # 2. 외부 데이터 로드
    oil = pd.read_csv(DATA_DIR / "oil_price.csv", parse_dates=["date"])
    currency = pd.read_csv(DATA_DIR / "currency.csv", parse_dates=["Date"])
    currency = currency.rename(columns={"Date": "date"})
    consumer_conf = pd.read_csv(DATA_DIR / "consumer_confidence.csv", parse_dates=["month"])
    marketing = pd.read_csv(DATA_DIR / "marketing_spend.csv", parse_dates=["date"])
    weather = pd.read_csv(DATA_DIR / "weather.csv", parse_dates=["date"])
    calendar = pd.read_csv(DATA_DIR / "calendar.csv", parse_dates=["date"])
    sku_meta = pd.read_csv(DATA_DIR / "sku_meta.csv", parse_dates=["launch_date"])
    ppt = pd.read_csv(DATA_DIR / "price_promo_train.csv", parse_dates=["date"])
    
    # 3. 기본 시간 피처
    demand["year"] = demand["date"].dt.year
    demand["month"] = demand["date"].dt.month
    demand["dayofyear"] = demand["date"].dt.dayofyear
    demand["weekday"] = demand["date"].dt.weekday
    demand["quarter"] = demand["date"].dt.quarter
    
    # 4. 계절성 피처 (EDA에서 발견: 9월 최고점, 1월 최저점)
    demand['month_sin'] = np.sin(2 * np.pi * demand['month'] / 12)
    demand['month_cos'] = np.cos(2 * np.pi * demand['month'] / 12)
    demand['dayofyear_sin'] = np.sin(2 * np.pi * demand['dayofyear'] / 365)
    demand['dayofyear_cos'] = np.cos(2 * np.pi * demand['dayofyear'] / 365)
    
    # 5. 계절 정보 추가
    demand = demand.merge(calendar[["date", "country", "season"]], on=["date", "country"], how="left")
    
    # 6. SKU 메타 정보
    demand = demand.merge(sku_meta[["sku", "family", "storage_gb", "launch_date"]], on="sku", how="left")
    demand["days_since_launch"] = (demand["date"] - demand["launch_date"]).dt.days.clip(lower=0)
    
    # 7. 시계열 피처 (EDA 기반 최적화)
    demand = demand.sort_values(["city", "sku", "date"])
    
    # 다양한 랙 피처 (EDA에서 중요도 확인)
    for lag in [1, 3, 7, 14, 30]:
        demand[f"demand_lag_{lag}"] = demand.groupby(["city", "sku"])["demand"].shift(lag).fillna(0)
    
    # 롤링 통계
    for window in [7, 14, 30]:
        demand[f"demand_rolling_mean_{window}"] = demand.groupby(["city", "sku"])["demand"].transform(
            lambda x: x.rolling(window=window, min_periods=1).mean()
        )
        demand[f"demand_rolling_std_{window}"] = demand.groupby(["city", "sku"])["demand"].transform(
            lambda x: x.rolling(window=window, min_periods=1).std()
        )
    
    # 8. 외부 요인 추가
    # 유가 데이터
    oil['pct_change'] = oil['brent_usd'].pct_change()
    oil['volatility_7d'] = oil['pct_change'].rolling(7).std()
    demand = demand.merge(oil[['date', 'brent_usd', 'pct_change', 'volatility_7d']], on='date', how='left')
    
    # 환율 데이터 (주요 환율만)
    fx_cols = ['EUR=X', 'KRW=X', 'JPY=X', 'GBP=X', 'CAD=X', 'AUD=X', 'BRL=X', 'ZAR=X']
    demand = demand.merge(currency[['date'] + fx_cols], on='date', how='left')
    
    # 소비자신뢰지수
    consumer_conf['year_month'] = consumer_conf['month'].dt.to_period('M')
    demand['year_month'] = demand['date'].dt.to_period('M')
    demand = demand.merge(consumer_conf[['year_month', 'country', 'confidence_index']], 
                         on=['year_month', 'country'], how='left')
    
    # 마케팅 지출
    demand = demand.merge(marketing[['date', 'country', 'spend_usd']], on=['date', 'country'], how='left')
    
    # 날씨 데이터
    demand = demand.merge(weather[['date', 'country', 'avg_temp', 'humidity']], on=['date', 'country'], how='left')
    
    # 결측치 처리
    demand = demand.fillna(0)
    
    # 9. 글로벌 신뢰지수 요인 생성 (출제자 방식)
    global_factor_df, pca, scaler = create_global_confidence_factor(consumer_conf)
    demand = demand.merge(global_factor_df, on='year_month', how='left')
    
    # 10. 가격 정보 추가
    demand = demand.merge(ppt[['date', 'sku', 'city', 'unit_price', 'discount_pct']], 
                         on=['date', 'sku', 'city'], how='left')
    
    # 11. 집계 피처 (EDA 기반)
    # 도시별 평균
    city_avg = demand.groupby('city')['demand'].mean().reset_index()
    city_avg = city_avg.rename(columns={'demand': 'city_avg_demand'})
    demand = demand.merge(city_avg, on='city', how='left')
    
    # SKU별 평균
    sku_avg = demand.groupby('sku')['demand'].mean().reset_index()
    sku_avg = sku_avg.rename(columns={'demand': 'sku_avg_demand'})
    demand = demand.merge(sku_avg, on='sku', how='left')
    
    # 국가별 평균
    country_avg = demand.groupby('country')['demand'].mean().reset_index()
    country_avg = country_avg.rename(columns={'demand': 'country_avg_demand'})
    demand = demand.merge(country_avg, on='country', how='left')
    
    # 12. 변동성 피처
    demand['demand_volatility'] = demand.groupby(['city', 'sku'])['demand'].transform(
        lambda x: x.rolling(window=30, min_periods=1).std()
    )
    
    # 13. 이벤트 탐지 및 피처 추가
    events_df = detect_events_using_zscore(demand, threshold=2.0)
    
    # 이벤트 플래그 추가
    demand['is_event_month'] = 0
    demand['event_multiplier'] = 1.0
    if len(events_df) > 0:
        for _, event in events_df.iterrows():
            mask = (demand['country'] == event['country']) & (demand['year_month'] == event['date'])
            demand.loc[mask, 'is_event_month'] = 1
            demand.loc[mask, 'event_multiplier'] = event['multiplier']
    
    print(f"데이터 로드 완료: {demand.shape}")
    print(f"이벤트 감지: {len(events_df)}개")
    
    return demand, events_df, pca, scaler

def train_enhanced_ensemble_model(train_data):
    """EDA 기반 고급 앙상블 모델 학습"""
    print("=== EDA 기반 고급 앙상블 모델 학습 ===")
    
    # 학습/검증 분할
    train_mask = train_data['year'] <= 2021
    val_mask = train_data['year'] == 2022
    
    train_set = train_data[train_mask].copy()
    val_set = train_data[val_mask].copy()
    
    # 피처 선택 (EDA 기반 최적화)
    feature_cols = [
        # 시간 피처
        'month', 'weekday', 'quarter', 'dayofyear',
        'month_sin', 'month_cos', 'dayofyear_sin', 'dayofyear_cos',
        
        # SKU 피처
        'days_since_launch', 'storage_gb',
        
        # 시계열 피처
        'demand_lag_1', 'demand_lag_3', 'demand_lag_7', 'demand_lag_14', 'demand_lag_30',
        'demand_rolling_mean_7', 'demand_rolling_mean_14', 'demand_rolling_mean_30',
        'demand_rolling_std_7', 'demand_rolling_std_14', 'demand_rolling_std_30',
        
        # 외부 요인
        'brent_usd', 'pct_change', 'volatility_7d',
        'confidence_index', 'spend_usd', 'avg_temp', 'humidity',
        'global_factor_1', 'global_factor_2',
        
        # 가격 정보
        'unit_price', 'discount_pct',
        
        # 집계 피처
        'city_avg_demand', 'sku_avg_demand', 'country_avg_demand',
        
        # 이벤트 피처
        'is_event_month', 'event_multiplier',
        
        # 변동성
        'demand_volatility'
    ]
    
    # 환율 피처 추가
    fx_cols = ['EUR=X', 'KRW=X', 'JPY=X', 'GBP=X', 'CAD=X', 'AUD=X', 'BRL=X', 'ZAR=X']
    feature_cols.extend(fx_cols)
    
    # 범주형 인코딩
    label_encoders = {}
    categorical_cols = ['city', 'sku', 'country', 'family', 'season']
    
    for col in categorical_cols:
        le = LabelEncoder()
        train_set[col + '_encoded'] = le.fit_transform(train_set[col].astype(str))
        val_set[col + '_encoded'] = le.transform(val_set[col].astype(str))
        label_encoders[col] = le
        feature_cols.append(col + '_encoded')
    
    # 결측치 처리
    for col in feature_cols:
        if col in train_set.columns:
            train_set[col] = train_set[col].fillna(0)
            val_set[col] = val_set[col].fillna(0)
    
    # 수치형 스케일링
    scaler = StandardScaler()
    numeric_features = [col for col in feature_cols if col not in [col + '_encoded' for col in categorical_cols]]
    
    X_train = train_set[feature_cols]
    X_train[numeric_features] = scaler.fit_transform(X_train[numeric_features])
    X_val = val_set[feature_cols]
    X_val[numeric_features] = scaler.transform(X_val[numeric_features])
    
    y_train = train_set['demand']
    y_val = val_set['demand']
    
    # 3가지 모델 학습 (EDA 기반 최적화)
    models = {
        'random_forest': RandomForestRegressor(
            n_estimators=200, max_depth=15, min_samples_split=10, 
            min_samples_leaf=5, random_state=42, n_jobs=-1
        ),
        'gradient_boosting': GradientBoostingRegressor(
            n_estimators=150, max_depth=8, learning_rate=0.1, 
            subsample=0.8, random_state=42
        ),
        'linear_regression': LinearRegression()
    }
    
    trained_models = {}
    predictions = {}
    
    for name, model in models.items():
        print(f"학습 중: {name}")
        model.fit(X_train, y_train)
        trained_models[name] = model
        
        # 검증 예측
        pred = model.predict(X_val)
        predictions[name] = pred
        
        # 개별 모델 성능
        rmse = np.sqrt(mean_squared_error(y_val, pred))
        r2 = r2_score(y_val, pred)
        print(f"  {name} - RMSE: {rmse:.2f}, R²: {r2:.3f}")
    
    # 동적 가중치 계산 (성능 기반)
    weights = {}
    total_score = 0
    for name, pred in predictions.items():
        rmse = np.sqrt(mean_squared_error(y_val, pred))
        score = 1 / (1 + rmse)  # RMSE가 낮을수록 높은 점수
        weights[name] = score
        total_score += score
    
    # 가중치 정규화
    for name in weights:
        weights[name] /= total_score
    
    print(f"\n동적 가중치: {weights}")
    
    # 앙상블 예측
    ensemble_pred = np.zeros(len(y_val))
    for name, pred in predictions.items():
        ensemble_pred += weights[name] * pred
    
    # 앙상블 성능
    ensemble_rmse = np.sqrt(mean_squared_error(y_val, ensemble_pred))
    ensemble_r2 = r2_score(y_val, ensemble_pred)
    print(f"\n앙상블 - RMSE: {ensemble_rmse:.2f}, R²: {ensemble_r2:.3f}")
    
    # 2022 검증 결과 시각화
    visualize_2022_validation(val_set, ensemble_pred, ensemble_r2)
    
    return trained_models, label_encoders, scaler, feature_cols, weights

def visualize_2022_validation(val_set, val_pred, r2_score):
    """2022 검증 결과 시각화"""
    print("=== 2022 검증 결과 시각화 ===")
    
    # val_pred를 val_set과 같은 순서로 정렬
    val_set_with_pred = val_set.copy()
    val_set_with_pred["predicted"] = val_pred
    
    # 일별 총 수요 집계
    daily_actual = val_set_with_pred.groupby("date")["demand"].sum().reset_index()
    daily_pred = val_set_with_pred.groupby("date")["predicted"].sum().reset_index()
    
    # 데이터 병합
    comparison = daily_actual.merge(daily_pred, on="date", suffixes=("", "_pred"))
    comparison = comparison.rename(columns={"predicted": "predicted"})
    
    # 시각화
    plt.figure(figsize=(15, 8))
    plt.plot(comparison["date"], comparison["demand"], label="Actual", alpha=0.8, linewidth=2, color='blue')
    plt.plot(comparison["date"], comparison["predicted"], label="Predicted", alpha=0.8, linewidth=2, color='red')
    plt.title(f"2022 Daily Demand: Actual vs Predicted (R² = {r2_score:.3f})")
    plt.xlabel("Date")
    plt.ylabel("Daily Total Demand")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # 성능 분석
    print(f"실제 평균: {comparison['demand'].mean():.1f}")
    print(f"예측 평균: {comparison['predicted'].mean():.1f}")
    print(f"편향: {comparison['predicted'].mean() - comparison['demand'].mean():.1f}")

def generate_enhanced_forecast():
    """EDA 기반 고급 예측 생성"""
    print("=== EDA 기반 고급 예측 생성 ===")
    
    # 1. 고급 데이터 로드
    train_data, events_df, pca, scaler = load_enhanced_training_data()
    
    # 2. 고급 모델 학습
    trained_models, label_encoders, scaler, feature_cols, weights = train_enhanced_ensemble_model(train_data)
    
    # 3. 미래 데이터 생성
    future_dates = pd.date_range(start="2023-01-01", end="2024-12-31", freq="D")
    future_df = pd.DataFrame({"date": future_dates})
    
    # 기본 피처 추가
    future_df["year"] = future_df["date"].dt.year
    future_df["month"] = future_df["date"].dt.month
    future_df["dayofyear"] = future_df["date"].dt.dayofyear
    future_df["weekday"] = future_df["date"].dt.weekday
    future_df["quarter"] = future_df["date"].dt.quarter
    
    # 계절성 피처
    future_df["month_sin"] = np.sin(2 * np.pi * future_df["month"] / 12)
    future_df["month_cos"] = np.cos(2 * np.pi * future_df["month"] / 12)
    future_df["dayofyear_sin"] = np.sin(2 * np.pi * future_df["dayofyear"] / 365)
    future_df["dayofyear_cos"] = np.cos(2 * np.pi * future_df["dayofyear"] / 365)
    
    # 외부 데이터 로드
    oil = pd.read_csv(DATA_DIR / "oil_price.csv", parse_dates=["date"])
    currency = pd.read_csv(DATA_DIR / "currency.csv", parse_dates=["Date"])
    currency = currency.rename(columns={"Date": "date"})
    consumer_conf = pd.read_csv(DATA_DIR / "consumer_confidence.csv", parse_dates=["month"])
    marketing = pd.read_csv(DATA_DIR / "marketing_spend.csv", parse_dates=["date"])
    weather = pd.read_csv(DATA_DIR / "weather.csv", parse_dates=["date"])
    calendar = pd.read_csv(DATA_DIR / "calendar.csv", parse_dates=["date"])
    sku_meta = pd.read_csv(DATA_DIR / "sku_meta.csv", parse_dates=["launch_date"])
    ppt = pd.read_csv(DATA_DIR / "price_promo_train.csv", parse_dates=["date"])
    
    # 미래 데이터에 외부 요인 추가
    oil['pct_change'] = oil['brent_usd'].pct_change()
    oil['volatility_7d'] = oil['pct_change'].rolling(7).std()
    future_df = future_df.merge(oil[['date', 'brent_usd', 'pct_change', 'volatility_7d']], on='date', how='left')
    
    fx_cols = ['EUR=X', 'KRW=X', 'JPY=X', 'GBP=X', 'CAD=X', 'AUD=X', 'BRL=X', 'ZAR=X']
    future_df = future_df.merge(currency[['date'] + fx_cols], on='date', how='left')
    
    future_df = future_df.merge(calendar[["date", "country", "season"]], on="date", how="left")
    future_df = future_df.merge(marketing[['date', 'country', 'spend_usd']], on=['date', 'country'], how='left')
    future_df = future_df.merge(weather[['date', 'country', 'avg_temp', 'humidity']], on=['date', 'country'], how='left')
    
    # 글로벌 요인 추가
    consumer_conf['year_month'] = consumer_conf['month'].dt.to_period('M')
    future_df['year_month'] = future_df['date'].dt.to_period('M')
    future_df = future_df.merge(consumer_conf[['year_month', 'country', 'confidence_index']], 
                               on=['year_month', 'country'], how='left')
    
    # 글로벌 요인 생성
    global_factor_df, _, _ = create_global_confidence_factor(consumer_conf)
    future_df = future_df.merge(global_factor_df, on='year_month', how='left')
    
    # 결측치 처리
    future_df = future_df.fillna(0)
    
    results = []
    
    # 각 도시-SKU 조합에 대해 예측
    cities = train_data["city"].unique()
    skus = train_data["sku"].unique()
    
    for city in cities:
        country = get_country_mapping()[city]
        
        for sku in skus:
            sku_info = sku_meta[sku_meta["sku"] == sku].iloc[0]
            
            # 해당 SKU의 과거 데이터
            sku_history = train_data[(train_data["city"] == city) & (train_data["sku"] == sku)].sort_values("date")
            
            if len(sku_history) == 0:
                continue
            
            # 시계열 데이터 초기화
            demand_series = sku_history["demand"].tolist()
            
            # 집계 값들 계산
            city_avg = sku_history["demand"].mean() if len(sku_history) > 0 else 0
            sku_avg = train_data[train_data["sku"] == sku]["demand"].mean()
            country_avg = train_data[train_data["country"] == country]["demand"].mean()
            
            # 각 미래 날짜에 대해 단계별 예측
            for date in future_dates:
                # 피처 생성
                date_features = future_df[future_df["date"] == date].copy()
                date_features["city"] = city
                date_features["country"] = country
                date_features["sku"] = sku
                date_features["family"] = sku_info["family"]
                date_features["storage_gb"] = sku_info["storage_gb"]
                date_features["launch_date"] = sku_info["launch_date"]
                date_features["days_since_launch"] = (date - date_features["launch_date"]).dt.days.clip(lower=0)
                
                # 시계열 피처 계산
                if len(demand_series) >= 30:
                    lag_1 = demand_series[-1]
                    lag_3 = demand_series[-3]
                    lag_7 = demand_series[-7]
                    lag_14 = demand_series[-14]
                    lag_30 = demand_series[-30]
                    rolling_mean_7 = np.mean(demand_series[-7:])
                    rolling_mean_14 = np.mean(demand_series[-14:])
                    rolling_mean_30 = np.mean(demand_series[-30:])
                    rolling_std_7 = np.std(demand_series[-7:])
                    rolling_std_14 = np.std(demand_series[-14:])
                    rolling_std_30 = np.std(demand_series[-30:])
                    volatility = np.std(demand_series[-30:])
                else:
                    recent_demand = demand_series[-1] if demand_series else 0
                    lag_1 = lag_3 = lag_7 = lag_14 = lag_30 = recent_demand
                    rolling_mean_7 = rolling_mean_14 = rolling_mean_30 = recent_demand
                    rolling_std_7 = rolling_std_14 = rolling_std_30 = 0
                    volatility = 0
                
                # 피처 설정
                date_features["demand_lag_1"] = lag_1
                date_features["demand_lag_3"] = lag_3
                date_features["demand_lag_7"] = lag_7
                date_features["demand_lag_14"] = lag_14
                date_features["demand_lag_30"] = lag_30
                date_features["demand_rolling_mean_7"] = rolling_mean_7
                date_features["demand_rolling_mean_14"] = rolling_mean_14
                date_features["demand_rolling_mean_30"] = rolling_mean_30
                date_features["demand_rolling_std_7"] = rolling_std_7
                date_features["demand_rolling_std_14"] = rolling_std_14
                date_features["demand_rolling_std_30"] = rolling_std_30
                date_features["city_avg_demand"] = city_avg
                date_features["sku_avg_demand"] = sku_avg
                date_features["country_avg_demand"] = country_avg
                date_features["demand_volatility"] = volatility
                
                # 가격 정보 추가
                sku_city_price = ppt[(ppt['sku'] == sku) & (ppt['city'] == city)]
                if len(sku_city_price) > 0:
                    date_features["unit_price"] = sku_city_price['unit_price'].mean()
                    date_features["discount_pct"] = sku_city_price['discount_pct'].mean()
                else:
                    date_features["unit_price"] = ppt['unit_price'].mean()
                    date_features["discount_pct"] = ppt['discount_pct'].mean()
                
                # 이벤트 확인
                date_features["is_event_month"] = 0
                date_features["event_multiplier"] = 1.0
                
                # 과거 이벤트 패턴 기반 동적 배수
                for _, event in events_df.iterrows():
                    if event['country'] == country:
                        if date.month == event['date'].month:
                            date_features["is_event_month"] = 1
                            date_features["event_multiplier"] = event['multiplier']
                            break
                
                # 범주형 인코딩
                for col in ["city", "sku", "country", "family", "season"]:
                    le = label_encoders[col]
                    date_features[col + "_encoded"] = le.transform(date_features[col].astype(str))
                
                # 스케일링
                X_pred = date_features[feature_cols]
                numeric_features = [col for col in feature_cols if col not in [col + '_encoded' for col in ["city", "sku", "country", "family", "season"]]]
                X_pred[numeric_features] = scaler.transform(X_pred[numeric_features])
                
                # 앙상블 예측
                ensemble_pred = 0
                for name, model in trained_models.items():
                    pred = model.predict(X_pred)[0]
                    ensemble_pred += weights[name] * pred
                
                # 이벤트 배수 적용
                final_pred = int(max(ensemble_pred * date_features["event_multiplier"].iloc[0], 0))
                
                results.append({
                    "date": date.strftime("%Y-%m-%d"),
                    "sku": sku,
                    "city": city,
                    "mean": final_pred
                })
                
                # 시계열 업데이트
                demand_series.append(final_pred)
                if len(demand_series) > 100:  # 메모리 효율성
                    demand_series = demand_series[-100:]
    
    # 결과 저장
    result_df = pd.DataFrame(results)
    output_path = DATA_DIR / "enhanced_forecast_submission.csv"
    result_df.to_csv(output_path, index=False)
    
    print(f"✅ EDA 기반 고급 예측 저장: {output_path}")
    print(f"총 예측 수: {len(result_df):,}")
    print(f"평균 수요: {result_df['mean'].mean():.1f}")
    print(f"최대 수요: {result_df['mean'].max():,}")
    print(f"최소 수요: {result_df['mean'].min()}")
    
    return result_df

def main():
    """메인 실행"""
    print("=== EDA 기반 고급 시계열 예측 모델 ===\n")
    
    result_df = generate_enhanced_forecast()
    
    print(f"\n✅ EDA 기반 고급 모델 완료!")
    print(f"📁 결과 파일: enhanced_forecast_submission.csv")

if __name__ == "__main__":
    main() 