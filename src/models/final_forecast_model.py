# src/models/final_forecast_model.py

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
    """학습 데이터 로드 (계절 정보 포함)"""
    # 수요 데이터
    conn = sqlite3.connect(DATA_DIR / "demand_train_processed.db")
    demand = pd.read_sql("SELECT * FROM demand_train", conn, parse_dates=['date'])
    conn.close()
    
    # 국가 매핑 추가
    country_map = get_country_mapping()
    demand["country"] = demand["city"].map(country_map)
    
    # 기본 피처들
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
    
    # 시계열 피처 - 더 다양한 랙과 롤링 윈도우
    demand = demand.sort_values(["city", "sku", "date"])
    
    # 다양한 랙 피처
    for lag in [1, 3, 7, 14, 30]:
        demand[f"demand_lag_{lag}"] = demand.groupby(["city", "sku"])["demand"].shift(lag).fillna(0)
    
    # 롤링 평균 피처 - 간단한 방법으로 변경
    for window in [7, 14, 30]:
        demand[f"demand_rolling_mean_{window}"] = demand.groupby(["city", "sku"])["demand"].transform(lambda x: x.rolling(window=window, min_periods=1).mean())
        demand[f"demand_rolling_std_{window}"] = demand.groupby(["city", "sku"])["demand"].transform(lambda x: x.rolling(window=window, min_periods=1).std())
    
    # 도시별, SKU별 평균 수요 (기준값)
    city_sku_avg = demand.groupby(["city", "sku"])["demand"].mean().reset_index()
    city_sku_avg = city_sku_avg.rename(columns={"demand": "city_sku_avg_demand"})
    demand = demand.merge(city_sku_avg, on=["city", "sku"], how="left")
    
    # 국가별 평균 수요
    country_avg = demand.groupby(["country", "sku"])["demand"].mean().reset_index()
    country_avg = country_avg.rename(columns={"demand": "country_sku_avg_demand"})
    demand = demand.merge(country_avg, on=["country", "sku"], how="left")
    
    # 월별 평균 수요
    month_avg = demand.groupby(["month", "sku"])["demand"].mean().reset_index()
    month_avg = month_avg.rename(columns={"demand": "month_sku_avg_demand"})
    demand = demand.merge(month_avg, on=["month", "sku"], how="left")
    
    return demand

def train_final_model(train_data):
    """최종 모델 학습"""
    train_mask = train_data["year"] <= 2021
    val_mask = train_data["year"] == 2022
    
    train_set = train_data[train_mask].copy()
    val_set = train_data[val_mask].copy()
    
    # 피처 선택 (season 사용, month 제거)
    feature_cols = [
        "city", "sku", "country", "family", 
        "season", "month", "quarter", "dayofyear", "weekday", "storage_gb",
        "days_since_launch", 
        "demand_lag_1", "demand_lag_3", "demand_lag_7", "demand_lag_14", "demand_lag_30",
        "demand_rolling_mean_7", "demand_rolling_mean_14", "demand_rolling_mean_30",
        "demand_rolling_std_7", "demand_rolling_std_14", "demand_rolling_std_30",
        "city_sku_avg_demand", "country_sku_avg_demand", "month_sku_avg_demand"
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
    
    # 모델 학습 - 과소평가 문제 해결을 위한 파라미터 조정
    X_train = train_set[feature_cols]
    y_train = train_set["demand"]
    
    model = RandomForestRegressor(
        n_estimators=200,  # 트리 수 증가
        max_depth=20,      # 깊이 증가
        min_samples_split=5,  # 분할 기준 완화
        min_samples_leaf=2,   # 리프 노드 최소 샘플 완화
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
    
    print(f"Final Model - Validation RMSE: {rmse:.2f}")
    print(f"Final Model - Validation R²: {r2:.3f}")
    
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
    plt.ylabel("Total Daily Demand")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    # 추가 통계 출력
    print(f"\n=== 2022 Validation Statistics ===")
    print(f"Total days: {len(comparison)}")
    print(f"Average actual demand: {comparison['demand'].mean():.1f}")
    print(f"Average predicted demand: {comparison['predicted'].mean():.1f}")
    errors = comparison["predicted"] - comparison["demand"]
    print(f"Mean absolute error: {np.abs(errors).mean():.1f}")
    print(f"Root mean squared error: {np.sqrt((errors**2).mean()):.1f}")

def generate_final_forecast():
    """최종 예측 생성"""
    
    print("Loading training data...")
    train_data = load_training_data()
    
    print("Training final model...")
    final_model, label_encoders, feature_cols = train_final_model(train_data)
    
    print("Loading forecast template...")
    template = pd.read_csv(DATA_DIR / "forecast_submission_template.csv", parse_dates=["date"])
    
    # 기본 피처 추가
    country_map = get_country_mapping()
    template["country"] = template["city"].map(country_map)
    template["year"] = template["date"].dt.year
    template["month"] = template["date"].dt.month
    template["quarter"] = template["date"].dt.quarter
    template["dayofyear"] = template["date"].dt.dayofyear
    template["weekday"] = template["date"].dt.weekday
    
    # 계절 정보 추가
    calendar = pd.read_csv(DATA_DIR / "calendar.csv", parse_dates=["date"])
    template = template.merge(calendar[["date", "country", "season"]], on=["date", "country"], how="left")
    
    # SKU 메타 정보
    sku_meta = pd.read_csv(DATA_DIR / "sku_meta.csv", parse_dates=["launch_date"])
    template = template.merge(sku_meta[["sku", "family", "storage_gb", "launch_date"]], on="sku", how="left")
    template["days_since_launch"] = (template["date"] - template["launch_date"]).dt.days.clip(lower=0)
    
    # 평균값 피처들 (2022년 데이터 기반)
    recent_data = train_data[train_data["year"] == 2022]
    
    # 도시별, SKU별 평균
    city_sku_avg = recent_data.groupby(["city", "sku"])["demand"].mean().reset_index()
    city_sku_avg = city_sku_avg.rename(columns={"demand": "city_sku_avg_demand"})
    template = template.merge(city_sku_avg, on=["city", "sku"], how="left")
    
    # 국가별, SKU별 평균
    country_sku_avg = recent_data.groupby(["country", "sku"])["demand"].mean().reset_index()
    country_sku_avg = country_sku_avg.rename(columns={"demand": "country_sku_avg_demand"})
    template = template.merge(country_sku_avg, on=["country", "sku"], how="left")
    
    # 월별, SKU별 평균
    month_sku_avg = recent_data.groupby(["month", "sku"])["demand"].mean().reset_index()
    month_sku_avg = month_sku_avg.rename(columns={"demand": "month_sku_avg_demand"})
    template = template.merge(month_sku_avg, on=["month", "sku"], how="left")
    
    # 랙 피처들 (최근 평균값으로 대체)
    recent_avg = recent_data.groupby(["city", "sku"])["demand"].mean()
    template = template.set_index(["city", "sku"]).join(recent_avg.rename("recent_avg")).reset_index()
    template["recent_avg"] = template["recent_avg"].fillna(train_data["demand"].mean())
    
    # 다양한 랙 피처들
    for lag in [1, 3, 7, 14, 30]:
        template[f"demand_lag_{lag}"] = template["recent_avg"] * (0.9 + 0.1 * np.random.random(len(template)))
    
    # 롤링 평균 피처들
    for window in [7, 14, 30]:
        template[f"demand_rolling_mean_{window}"] = template["recent_avg"] * (0.95 + 0.05 * np.random.random(len(template)))
        template[f"demand_rolling_std_{window}"] = template["recent_avg"] * 0.1
    
    # 누락값 처리
    for col in ["city_sku_avg_demand", "country_sku_avg_demand", "month_sku_avg_demand"]:
        template[col] = template[col].fillna(template["recent_avg"])
    
    # 범주형 인코딩
    for col, le in label_encoders.items():
        try:
            template[col + "_encoded"] = le.transform(template[col].astype(str))
        except ValueError:
            template[col + "_encoded"] = 0
    
    # 기본 예측
    print("Generating base predictions...")
    X_pred = template[feature_cols]
    base_predictions = final_model.predict(X_pred)
    
    # 이벤트 배수 적용
    template["base_pred"] = base_predictions
    template["event_multiplier"] = 1.0
    
    event_multipliers = get_event_multipliers()
    
    print("Applying event multipliers...")
    for year, year_events in event_multipliers.items():
        for country, (start_date, end_date, multiplier) in year_events.items():
            mask = (
                (template["country"] == country) &
                (template["date"] >= start_date) &
                (template["date"] <= end_date)
            )
            template.loc[mask, "event_multiplier"] = multiplier
    
    # 최종 예측값 계산
    template["final_pred"] = template["base_pred"] * template["event_multiplier"]
    template["mean"] = np.maximum(0, template["final_pred"].round().astype(int))
    
    # 결과 정리
    result = template[["date", "sku", "city", "mean"]].copy()
    
    return result, template

def main():
    """메인 실행"""
    print("=== Final Forecast Model ===\n")
    
    # 예측 생성
    forecast, detailed_template = generate_final_forecast()
    
    # 결과 저장
    output_path = DATA_DIR / "final_forecast_submission.csv"
    forecast.to_csv(output_path, index=False)
    print(f"\nForecast saved to: {output_path}")
    
    # 결과 요약
    print(f"\n=== Forecast Summary ===")
    print(f"Total predictions: {len(forecast):,}")
    print(f"Date range: {forecast['date'].min()} to {forecast['date'].max()}")
    print(f"Average daily demand: {forecast['mean'].mean():.1f}")
    print(f"Max daily demand: {forecast['mean'].max():,}")
    print(f"Min daily demand: {forecast['mean'].min()}")
    
    # 이벤트 기간 분석
    country_map = get_country_mapping()
    detailed_template["country"] = detailed_template["city"].map(country_map)
    event_multipliers = get_event_multipliers()
    
    print(f"\n=== Event Period Analysis ===")
    
    for year, year_events in event_multipliers.items():
        print(f"\n{year} Events:")
        for country, (start_date, end_date, multiplier) in year_events.items():
            event_mask = (
                (detailed_template["country"] == country) &
                (detailed_template["date"] >= start_date) &
                (detailed_template["date"] <= end_date)
            )
            event_data = detailed_template[event_mask]
            
            if len(event_data) > 0:
                avg_demand = event_data["mean"].mean()
                print(f"  {country} ({start_date} to {end_date}):")
                print(f"    Multiplier: {multiplier:.2f}x")
                print(f"    Avg Demand: {avg_demand:.1f}")
    
    # 계절별 분석
    print(f"\n=== Seasonal Analysis ===")
    seasonal_stats = detailed_template.groupby("season")["mean"].agg(["mean", "std", "count"]).round(2)
    print("Demand by season:")
    print(seasonal_stats)
    
    return forecast

if __name__ == "__main__":
    forecast = main() 