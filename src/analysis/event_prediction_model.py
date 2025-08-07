# src/analysis/event_prediction_model.py

import pandas as pd
import numpy as np
import sqlite3
from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# === 경로 설정 ===
SCRIPT = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT.parents[2]
DATA_DIR = PROJECT_ROOT / "data"

def load_all_data():
    """모든 데이터 로드"""
    # 수요 데이터
    conn = sqlite3.connect(DATA_DIR / "demand_train.db")
    demand = pd.read_sql("SELECT * FROM demand_train", conn, parse_dates=['date'])
    conn.close()
    
    # 외부 요인 데이터
    oil = pd.read_csv(DATA_DIR / "oil_price_processed.csv", parse_dates=["date"])
    currency = pd.read_csv(DATA_DIR / "currency_processed.csv", parse_dates=["date"])
    consumer_conf = pd.read_csv(DATA_DIR / "consumer_confidence_processed.csv", parse_dates=["date"])
    marketing = pd.read_csv(DATA_DIR / "marketing_spend.csv", parse_dates=["date"])
    
    return demand, oil, currency, consumer_conf, marketing

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

def create_event_labels(demand_df: pd.DataFrame, threshold_percentile: float = 95):
    """과거 이벤트를 라벨로 생성"""
    country_map = get_country_mapping()
    demand_df = demand_df.copy()
    demand_df["country"] = demand_df["city"].map(country_map)
    
    # 월별 국가 수요 집계
    monthly_country = (
        demand_df.groupby(["country", demand_df["date"].dt.to_period("M")])["demand"]
        .sum()
        .reset_index()
    )
    monthly_country.columns = ["country", "year_month", "demand"]
    
    # Z-score 계산
    monthly_country["z_score"] = (
        monthly_country.groupby("country")["demand"]
        .transform(lambda x: (x - x.mean()) / x.std())
    )
    
    # 이벤트 플래그 생성
    threshold = np.percentile(monthly_country["z_score"].dropna(), threshold_percentile)
    monthly_country["is_event"] = (monthly_country["z_score"] >= threshold).astype(int)
    
    return monthly_country

def build_prediction_features(oil_df, consumer_df, marketing_df, currency_df):
    """예측을 위한 피처 구축"""
    
    # 날짜 범위: 2018-01 부터 2024-12까지
    date_range = pd.date_range("2018-01-01", "2024-12-31", freq="MS")
    countries = ['USA', 'DEU', 'FRA', 'KOR', 'JPN', 'GBR', 'CAN', 'AUS', 'BRA', 'ZAF']
    
    # 기본 프레임 생성
    base_df = []
    for date in date_range:
        for country in countries:
            base_df.append({
                "date": date,
                "country": country,
                "year": date.year,
                "month": date.month,
                "year_month": date.to_period("M")
            })
    
    features_df = pd.DataFrame(base_df)
    
    # === 1. 시간 기반 피처 ===
    features_df["season"] = features_df["month"].map({
        12: "winter", 1: "winter", 2: "winter",
        3: "spring", 4: "spring", 5: "spring", 
        6: "summer", 7: "summer", 8: "summer",
        9: "autumn", 10: "autumn", 11: "autumn"
    })
    
    features_df["is_peak_month"] = features_df["month"].isin([7, 8, 11, 12]).astype(int)
    features_df["is_mid_year"] = features_df["month"].isin([6, 7, 8]).astype(int)
    features_df["is_year_end"] = features_df["month"].isin([10, 11, 12]).astype(int)
    
    # === 2. 국가별 이벤트 패턴 ===
    # 과거 이벤트 빈도 (2018-2022 기준)
    historical_freq = {
        'BRA': 5, 'DEU': 4, 'AUS': 3, 'CAN': 3, 'GBR': 3, 
        'FRA': 3, 'JPN': 3, 'USA': 2, 'KOR': 2, 'ZAF': 2
    }
    features_df["historical_event_freq"] = features_df["country"].map(historical_freq)
    
    # 마지막 이벤트 이후 경과 시간
    last_events = {
        'USA': '2021-05', 'KOR': '2022-03', 'AUS': '2022-12', 'BRA': '2022-12',
        'CAN': '2022-11', 'GBR': '2022-12', 'FRA': '2022-10', 'ZAF': '2022-12',
        'JPN': '2022-09', 'DEU': '2022-12'
    }
    
    for country, last_event in last_events.items():
        mask = features_df["country"] == country
        last_period = pd.Period(last_event)
        features_df.loc[mask, "months_since_last_event"] = (
            features_df.loc[mask, "year_month"].apply(lambda x: (x - last_period).n if pd.notna(x) else 0)
        )
    
    # === 3. 외부 요인 피처 ===
    # 유가 데이터 병합 (월별 집계)
    oil_monthly = oil_df.copy()
    oil_monthly["year_month"] = oil_monthly["date"].dt.to_period("M")
    oil_agg = (
        oil_monthly.groupby("year_month")
        .agg({
            "brent_usd": ["mean", "std"],
            "oil_spike": "sum",
            "oil_rise": "sum", 
            "oil_fall": "sum"
        })
        .round(2)
    )
    oil_agg = oil_agg.reset_index()
    # 컬럼명 수동 설정
    oil_agg.columns = ["year_month", "oil_mean", "oil_std", "oil_spike_sum", "oil_rise_sum", "oil_fall_sum"]
    
    features_df = features_df.merge(oil_agg, on="year_month", how="left")
    
    # 소비자 신뢰지수 병합
    consumer_monthly = consumer_df.copy()
    consumer_monthly["year_month"] = consumer_monthly["date"].dt.to_period("M")
    consumer_agg = (
        consumer_monthly.groupby(["country", "year_month"])["confidence_index"]
        .mean()
        .reset_index()
    )
    
    features_df = features_df.merge(consumer_agg, on=["country", "year_month"], how="left")
    
    # 마케팅 비용 병합
    marketing_monthly = marketing_df.copy()
    marketing_monthly["year_month"] = marketing_monthly["date"].dt.to_period("M")
    marketing_agg = (
        marketing_monthly.groupby(["country", "year_month"])["spend_usd"]
        .sum()
        .reset_index()
    )
    
    features_df = features_df.merge(marketing_agg, on=["country", "year_month"], how="left")
    
    # === 4. 상호작용 피처 ===
    features_df["confidence_x_oil"] = features_df["confidence_index"] * features_df["oil_mean"]
    features_df["marketing_x_month"] = features_df["spend_usd"] * features_df["is_peak_month"]
    
    # === 5. 결측치 처리 ===
    # 2023-2024년 데이터는 결측이므로 외삽/보간
    
    # 유가는 2022년 평균값 사용
    oil_2022_mean = features_df[features_df["year"] == 2022]["oil_mean"].mean()
    features_df["oil_mean"].fillna(oil_2022_mean, inplace=True)
    features_df["oil_std"].fillna(0, inplace=True)
    features_df["oil_spike_sum"].fillna(0, inplace=True)
    features_df["oil_rise_sum"].fillna(0, inplace=True)
    features_df["oil_fall_sum"].fillna(0, inplace=True)
    
    # 소비자 신뢰지수는 국가별 2022년 평균값 사용
    for country in countries:
        country_mask = features_df["country"] == country
        recent_confidence = features_df[
            (features_df["country"] == country) & 
            (features_df["year"] == 2022)
        ]["confidence_index"].mean()
        
        if pd.isna(recent_confidence):
            recent_confidence = 100  # 기본값
            
        features_df.loc[country_mask, "confidence_index"] = (
            features_df.loc[country_mask, "confidence_index"].fillna(recent_confidence)
        )
    
    # 마케팅 비용은 0으로 채움
    features_df["spend_usd"].fillna(0, inplace=True)
    
    # 상호작용 피처 재계산
    features_df["confidence_x_oil"] = features_df["confidence_index"] * features_df["oil_mean"]
    features_df["marketing_x_month"] = features_df["spend_usd"] * features_df["is_peak_month"]
    
    return features_df

def train_event_prediction_model(features_df, event_labels_df):
    """이벤트 예측 모델 학습"""
    
    # 라벨 병합
    train_df = features_df.merge(
        event_labels_df[["country", "year_month", "is_event"]], 
        on=["country", "year_month"], 
        how="left"
    )
    
    # 학습 데이터는 2018-2022년만
    train_mask = train_df["year"] <= 2022
    train_data = train_df[train_mask].copy()
    train_data["is_event"].fillna(0, inplace=True)
    
    # 피처 선택
    feature_cols = [
        "month", "is_peak_month", "is_mid_year", "is_year_end",
        "historical_event_freq", "months_since_last_event",
        "oil_mean", "oil_std", "oil_spike_sum", "oil_rise_sum", "oil_fall_sum",
        "confidence_index", "spend_usd",
        "confidence_x_oil", "marketing_x_month"
    ]
    
    # 결측치 처리
    for col in feature_cols:
        if col in train_data.columns:
            train_data[col].fillna(0, inplace=True)
    
    X = train_data[feature_cols]
    y = train_data["is_event"]
    
    # 국가별 원핫 인코딩 추가
    country_dummies = pd.get_dummies(train_data["country"], prefix="country")
    X = pd.concat([X, country_dummies], axis=1)
    
    # 모델 학습
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        random_state=42,
        class_weight="balanced"  # 불균형 데이터 처리
    )
    
    model.fit(X, y)
    
    # 피처 중요도
    feature_importance = pd.DataFrame({
        "feature": X.columns,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False)
    
    print("=== Feature Importance ===")
    print(feature_importance.head(10))
    
    return model, feature_cols, feature_importance

def predict_2023_2024_events(model, features_df, feature_cols):
    """2023-2024년 이벤트 예측"""
    
    # 예측 데이터 준비
    predict_mask = features_df["year"].isin([2023, 2024])
    predict_data = features_df[predict_mask].copy()
    
    # 피처 준비
    for col in feature_cols:
        if col in predict_data.columns:
            predict_data[col].fillna(0, inplace=True)
    
    X_pred = predict_data[feature_cols]
    
    # 국가별 원핫 인코딩
    country_dummies = pd.get_dummies(predict_data["country"], prefix="country")
    
    # 학습 시와 동일한 컬럼 확보
    for col in model.feature_names_in_:
        if col not in X_pred.columns and col not in country_dummies.columns:
            X_pred[col] = 0
        elif col in country_dummies.columns:
            X_pred = pd.concat([X_pred, country_dummies[[col]]], axis=1)
    
    # 컬럼 순서 맞춤
    X_pred = X_pred.reindex(columns=model.feature_names_in_, fill_value=0)
    
    # 예측
    probabilities = model.predict_proba(X_pred)[:, 1]  # 이벤트 확률
    predictions = model.predict(X_pred)
    
    # 결과 정리
    predict_data["event_probability"] = probabilities
    predict_data["predicted_event"] = predictions
    
    return predict_data

def analyze_predictions(predictions_df):
    """예측 결과 분석"""
    
    # 2023년 예측
    pred_2023 = predictions_df[predictions_df["year"] == 2023]
    top_2023 = pred_2023.nlargest(10, "event_probability")[
        ["country", "month", "year_month", "event_probability", "predicted_event"]
    ]
    
    print("\n=== 2023 Top Event Predictions ===")
    print(top_2023.to_string(index=False))
    
    # 2024년 예측  
    pred_2024 = predictions_df[predictions_df["year"] == 2024]
    top_2024 = pred_2024.nlargest(10, "event_probability")[
        ["country", "month", "year_month", "event_probability", "predicted_event"]
    ]
    
    print("\n=== 2024 Top Event Predictions ===")
    print(top_2024.to_string(index=False))
    
    # 국가별 최고 확률 월
    best_by_country = (
        predictions_df.loc[predictions_df.groupby(["country", "year"])["event_probability"].idxmax()]
        [["country", "year", "month", "year_month", "event_probability"]]
        .sort_values(["year", "event_probability"], ascending=[True, False])
    )
    
    print("\n=== Best Month by Country and Year ===")
    print(best_by_country.to_string(index=False))
    
    return top_2023, top_2024, best_by_country

def visualize_predictions(predictions_df):
    """예측 결과 시각화"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. 2023년 국가별 월평균 확률
    pred_2023 = predictions_df[predictions_df["year"] == 2023]
    country_prob_2023 = pred_2023.groupby("country")["event_probability"].mean()
    country_prob_2023.plot(kind="bar", ax=axes[0,0], title="2023 Average Event Probability by Country")
    
    # 2. 2024년 국가별 월평균 확률
    pred_2024 = predictions_df[predictions_df["year"] == 2024]
    country_prob_2024 = pred_2024.groupby("country")["event_probability"].mean()
    country_prob_2024.plot(kind="bar", ax=axes[0,1], title="2024 Average Event Probability by Country")
    
    # 3. 월별 평균 확률 (2023-2024)
    month_prob = predictions_df.groupby("month")["event_probability"].mean()
    month_prob.plot(kind="bar", ax=axes[1,0], title="Average Event Probability by Month")
    
    # 4. 시간에 따른 확률 변화 (상위 5개 국가)
    top_countries = predictions_df.groupby("country")["event_probability"].mean().nlargest(5).index
    
    for country in top_countries:
        country_data = predictions_df[predictions_df["country"] == country]
        axes[1,1].plot(range(len(country_data)), country_data["event_probability"], 
                      label=country, marker='o', markersize=3)
    
    axes[1,1].set_title("Event Probability Timeline (Top 5 Countries)")
    axes[1,1].set_xlabel("Month (2023-2024)")
    axes[1,1].legend()
    
    plt.tight_layout()
    plt.show()

def main():
    """메인 실행"""
    print("=== Event Prediction Model ===\n")
    
    # 데이터 로드
    demand, oil, currency, consumer_conf, marketing = load_all_data()
    
    # 이벤트 라벨 생성
    event_labels = create_event_labels(demand)
    print(f"Created event labels: {event_labels['is_event'].sum()} events out of {len(event_labels)} months")
    
    # 피처 구축
    features = build_prediction_features(oil, consumer_conf, marketing, currency)
    print(f"Built features: {len(features)} country-month combinations")
    print(f"Feature columns: {list(features.columns)}")
    
    # 모델 학습
    model, feature_cols, importance = train_event_prediction_model(features, event_labels)
    
    # 예측
    predictions = predict_2023_2024_events(model, features, feature_cols)
    
    # 결과 분석
    top_2023, top_2024, best_by_country = analyze_predictions(predictions)
    
    # 시각화
    visualize_predictions(predictions)
    
    return model, predictions, importance

if __name__ == "__main__":
    model, predictions, importance = main()