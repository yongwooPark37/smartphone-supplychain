# src/analysis/analyze_historical_events.py

import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple
import seaborn as sns

# === 경로 설정 ===
SCRIPT = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT.parents[2]
DATA_DIR = PROJECT_ROOT / "data"

def load_data():
    """모든 필요한 데이터 로드"""
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
    """국가 매핑 정보"""
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

def detect_major_events(demand_df: pd.DataFrame, threshold_percentile: float = 95) -> pd.DataFrame:
    """
    각 국가별로 급등 이벤트를 탐지합니다.
    매년 한 국가에서 1~3개월 지속되는 이벤트를 찾습니다.
    """
    country_map = get_country_mapping()
    demand_df = demand_df.copy()
    demand_df["country"] = demand_df["city"].map(country_map)
    
    # 국가별 일별 수요 집계
    daily_country = (
        demand_df.groupby(["country", "date"])["demand"]
        .sum()
        .reset_index()
    )
    
    # 월별 국가 수요 집계
    daily_country["year"] = daily_country["date"].dt.year
    daily_country["month"] = daily_country["date"].dt.month
    daily_country["year_month"] = daily_country["date"].dt.to_period("M")
    
    monthly_country = (
        daily_country.groupby(["country", "year_month"])["demand"]
        .sum()
        .reset_index()
    )
    
    # 국가별 Z-score 계산
    monthly_country["z_score"] = (
        monthly_country.groupby("country")["demand"]
        .transform(lambda x: (x - x.mean()) / x.std())
    )
    
    # 상위 percentile 이벤트 추출
    threshold = np.percentile(monthly_country["z_score"].dropna(), threshold_percentile)
    major_events = monthly_country[monthly_country["z_score"] >= threshold].copy()
    
    # 연도별 정렬
    major_events["year"] = major_events["year_month"].dt.year
    major_events = major_events.sort_values(["year", "z_score"], ascending=[True, False])
    
    return major_events

def analyze_event_patterns(events_df: pd.DataFrame) -> Dict:
    """이벤트 패턴 분석"""
    
    # 1. 연도별 이벤트 국가 분포
    yearly_countries = events_df.groupby("year")["country"].apply(list).to_dict()
    
    # 2. 이벤트 시기 (월) 분포
    month_distribution = events_df["year_month"].dt.month.value_counts().sort_index()
    
    # 3. 국가별 이벤트 빈도
    country_frequency = events_df["country"].value_counts()
    
    # 4. 연속성 분석 - 같은 국가에서 연속된 월에 이벤트가 있는지
    continuity_analysis = {}
    for country in events_df["country"].unique():
        country_events = events_df[events_df["country"] == country].sort_values("year_month")
        periods = []
        
        if len(country_events) > 0:
            current_start = country_events.iloc[0]["year_month"]
            current_end = current_start
            
            for i in range(1, len(country_events)):
                current_period = country_events.iloc[i]["year_month"]
                prev_period = country_events.iloc[i-1]["year_month"]
                
                # 연속된 월인지 확인
                if (current_period - prev_period).n == 1:
                    current_end = current_period
                else:
                    periods.append((current_start, current_end))
                    current_start = current_period
                    current_end = current_period
            
            periods.append((current_start, current_end))
            continuity_analysis[country] = periods
    
    return {
        "yearly_countries": yearly_countries,
        "month_distribution": month_distribution,
        "country_frequency": country_frequency,
        "continuity_analysis": continuity_analysis
    }

def correlate_with_external_factors(events_df: pd.DataFrame, oil_df: pd.DataFrame, 
                                  marketing_df: pd.DataFrame, consumer_df: pd.DataFrame) -> Dict:
    """외부 요인과의 상관관계 분석"""
    
    # 이벤트 월을 시계열로 확장
    event_flags = []
    for _, row in events_df.iterrows():
        period = row["year_month"]
        start_date = period.to_timestamp()
        end_date = period.to_timestamp() + pd.offsets.MonthEnd(0)
        dates = pd.date_range(start_date, end_date, freq="D")
        
        for date in dates:
            event_flags.append({
                "date": date,
                "country": row["country"],
                "event_flag": 1,
                "z_score": row["z_score"]
            })
    
    event_daily = pd.DataFrame(event_flags)
    
    # 외부 요인들과 병합
    correlation_data = event_daily.copy()
    
    # 유가
    correlation_data = correlation_data.merge(
        oil_df[["date", "brent_usd", "oil_spike", "oil_rise", "oil_fall"]], 
        on="date", how="left"
    )
    
    # 마케팅 비용
    correlation_data = correlation_data.merge(
        marketing_df[["date", "country", "spend_usd"]], 
        on=["date", "country"], how="left"
    )
    
    # 소비자 신뢰지수
    correlation_data = correlation_data.merge(
        consumer_df[["date", "country", "confidence_index"]], 
        on=["date", "country"], how="left"
    )
    
    # 상관계수 계산
    correlations = {}
    
    # 숫자형 컬럼들과의 상관계수
    numeric_cols = ["brent_usd", "spend_usd", "confidence_index"]
    for col in numeric_cols:
        if col in correlation_data.columns:
            valid_data = correlation_data.dropna(subset=[col, "z_score"])
            if len(valid_data) > 0:
                corr = valid_data["z_score"].corr(valid_data[col])
                correlations[col] = corr
    
    # 유가 스파이크와의 관계
    oil_spike_relation = {}
    for country in events_df["country"].unique():
        country_events = correlation_data[correlation_data["country"] == country]
        if len(country_events) > 0:
            spike_during_event = country_events["oil_spike"].sum()
            total_event_days = len(country_events)
            oil_spike_relation[country] = spike_during_event / total_event_days if total_event_days > 0 else 0
    
    return {
        "correlations": correlations,
        "oil_spike_relation": oil_spike_relation,
        "correlation_data": correlation_data
    }

def visualize_analysis(events_df: pd.DataFrame, patterns: Dict, correlations: Dict):
    """분석 결과 시각화"""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. 연도별 이벤트 국가
    yearly_data = []
    for year, countries in patterns["yearly_countries"].items():
        for country in countries:
            yearly_data.append({"year": year, "country": country})
    
    yearly_df = pd.DataFrame(yearly_data)
    if len(yearly_df) > 0:
        yearly_pivot = yearly_df.groupby(["year", "country"]).size().unstack(fill_value=0)
        yearly_pivot.plot(kind="bar", ax=axes[0,0], stacked=True)
        axes[0,0].set_title("Events by Year and Country")
        axes[0,0].set_xlabel("Year")
        axes[0,0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # 2. 월별 이벤트 분포
    patterns["month_distribution"].plot(kind="bar", ax=axes[0,1])
    axes[0,1].set_title("Event Distribution by Month")
    axes[0,1].set_xlabel("Month")
    
    # 3. 국가별 이벤트 빈도
    patterns["country_frequency"].plot(kind="bar", ax=axes[0,2])
    axes[0,2].set_title("Event Frequency by Country")
    axes[0,2].set_xlabel("Country")
    
    # 4. Z-score 시계열
    events_df_sorted = events_df.sort_values("year_month")
    axes[1,0].plot(events_df_sorted["year_month"].astype(str), events_df_sorted["z_score"], 'o-')
    axes[1,0].set_title("Event Intensity (Z-score) Over Time")
    axes[1,0].tick_params(axis='x', rotation=45)
    
    # 5. 외부 요인 상관계수
    if correlations["correlations"]:
        corr_df = pd.Series(correlations["correlations"])
        corr_df.plot(kind="bar", ax=axes[1,1])
        axes[1,1].set_title("Correlation with External Factors")
        axes[1,1].set_ylabel("Correlation Coefficient")
    
    # 6. 유가 스파이크 관계
    if correlations["oil_spike_relation"]:
        oil_rel_df = pd.Series(correlations["oil_spike_relation"])
        oil_rel_df.plot(kind="bar", ax=axes[1,2])
        axes[1,2].set_title("Oil Spike Ratio During Events")
        axes[1,2].set_ylabel("Spike Days / Event Days")
    
    plt.tight_layout()
    plt.show()
    
    return fig

def predict_2023_2024_events(patterns: Dict, correlations: Dict, 
                           oil_df: pd.DataFrame, marketing_df: pd.DataFrame) -> Dict:
    """2023-2024년 이벤트 예측"""
    
    predictions = {}
    
    # 1. 과거 패턴 기반 예측
    yearly_countries = patterns["yearly_countries"]
    
    # 각 국가별 마지막 이벤트 연도 계산
    country_last_event = {}
    for year, countries in yearly_countries.items():
        for country in countries:
            if country not in country_last_event or year > country_last_event[country]:
                country_last_event[country] = year
    
    # 2023, 2024년 후보 국가들 (오랫동안 이벤트가 없었던 국가들 우선)
    all_countries = ['USA', 'DEU', 'FRA', 'KOR', 'JPN', 'GBR', 'CAN', 'AUS', 'BRA', 'ZAF']
    candidate_countries = []
    
    for country in all_countries:
        last_event = country_last_event.get(country, 2017)  # 기본값은 2017
        gap = 2023 - last_event
        candidate_countries.append((country, gap))
    
    # 갭이 큰 순서로 정렬
    candidate_countries.sort(key=lambda x: x[1], reverse=True)
    
    # 월별 패턴 (가장 빈번한 월들)
    top_months = patterns["month_distribution"].head(3).index.tolist()
    
    # 예측 결과
    predictions["2023"] = {
        "primary_candidate": candidate_countries[0][0],
        "secondary_candidates": [c[0] for c in candidate_countries[1:3]],
        "likely_months": top_months,
        "confidence": "Medium - based on historical gaps"
    }
    
    predictions["2024"] = {
        "primary_candidate": candidate_countries[1][0],
        "secondary_candidates": [c[0] for c in candidate_countries[2:4]],
        "likely_months": top_months,
        "confidence": "Low - further from training data"
    }
    
    return predictions

def main():
    """메인 분석 실행"""
    print("=== Historical Event Analysis ===\n")
    
    # 데이터 로드
    demand, oil, currency, consumer_conf, marketing = load_data()
    
    # 주요 이벤트 탐지
    events = detect_major_events(demand, threshold_percentile=95)
    print(f"Detected {len(events)} major events:")
    print(events[["country", "year_month", "z_score", "demand"]].to_string())
    print()
    
    # 패턴 분석
    patterns = analyze_event_patterns(events)
    
    print("=== Event Patterns ===")
    print("Yearly distribution:")
    for year, countries in patterns["yearly_countries"].items():
        print(f"  {year}: {countries}")
    
    print(f"\nMonth distribution:\n{patterns['month_distribution']}")
    print(f"\nCountry frequency:\n{patterns['country_frequency']}")
    print()
    
    # 외부 요인과의 상관관계
    correlations = correlate_with_external_factors(events, oil, marketing, consumer_conf)
    
    print("=== Correlations with External Factors ===")
    for factor, corr in correlations["correlations"].items():
        print(f"{factor}: {corr:.3f}")
    
    print("\nOil spike ratios during events:")
    for country, ratio in correlations["oil_spike_relation"].items():
        print(f"  {country}: {ratio:.3f}")
    print()
    
    # 시각화
    visualize_analysis(events, patterns, correlations)
    
    # 2023-2024 예측
    predictions = predict_2023_2024_events(patterns, correlations, oil, marketing)
    
    print("=== 2023-2024 Event Predictions ===")
    for year, pred in predictions.items():
        print(f"{year}:")
        print(f"  Primary candidate: {pred['primary_candidate']}")
        print(f"  Secondary candidates: {pred['secondary_candidates']}")
        print(f"  Likely months: {pred['likely_months']}")
        print(f"  Confidence: {pred['confidence']}")
    print()
    
    return events, patterns, correlations, predictions

if __name__ == "__main__":
    events, patterns, correlations, predictions = main()