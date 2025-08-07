# src/analysis/analyze_event_multipliers.py

import pandas as pd
import numpy as np
import sqlite3
from pathlib import Path
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

def load_historical_data():
    """과거 데이터 로드"""
    # 수요 데이터
    conn = sqlite3.connect(DATA_DIR / "demand_train.db")
    demand = pd.read_sql("SELECT * FROM demand_train", conn, parse_dates=['date'])
    conn.close()
    
    # 국가 매핑 추가
    country_map = get_country_mapping()
    demand["country"] = demand["city"].map(country_map)
    
    return demand

def detect_historical_events(demand_df, threshold_percentile=95):
    """과거 이벤트 탐지"""
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

def calculate_event_multipliers(demand_df, event_df):
    """이벤트 기간의 실제 수요 배수 계산"""
    
    # 이벤트가 발생한 월들 찾기
    event_months = event_df[event_df["is_event"] == 1].copy()
    
    multipliers = []
    
    for _, event_row in event_months.iterrows():
        country = event_row["country"]
        event_month = event_row["year_month"]
        
        # 해당 국가의 전체 데이터
        country_data = demand_df[demand_df["country"] == country].copy()
        country_data["year_month"] = country_data["date"].dt.to_period("M")
        
        # 이벤트 월 데이터
        event_month_data = country_data[country_data["year_month"] == event_month]
        
        # 이벤트 월의 평균 수요
        event_avg_demand = event_month_data["demand"].mean()
        
        # 해당 국가의 전체 평균 수요 (이벤트 월 제외)
        non_event_data = country_data[country_data["year_month"] != event_month]
        baseline_avg_demand = non_event_data["demand"].mean()
        
        # 배수 계산
        if baseline_avg_demand > 0:
            multiplier = event_avg_demand / baseline_avg_demand
            multipliers.append({
                "country": country,
                "event_month": event_month,
                "event_avg_demand": event_avg_demand,
                "baseline_avg_demand": baseline_avg_demand,
                "multiplier": multiplier,
                "z_score": event_row["z_score"]
            })
    
    return pd.DataFrame(multipliers)

def analyze_multiplier_patterns(multipliers_df):
    """배수 패턴 분석"""
    
    print("=== Historical Event Multiplier Analysis ===\n")
    
    # 전체 통계
    print("Overall Statistics:")
    print(f"Total events analyzed: {len(multipliers_df)}")
    print(f"Average multiplier: {multipliers_df['multiplier'].mean():.2f}x")
    print(f"Median multiplier: {multipliers_df['multiplier'].median():.2f}x")
    print(f"Min multiplier: {multipliers_df['multiplier'].min():.2f}x")
    print(f"Max multiplier: {multipliers_df['multiplier'].max():.2f}x")
    print(f"Standard deviation: {multipliers_df['multiplier'].std():.2f}x")
    
    # 국가별 분석
    print("\n=== Country-wise Analysis ===")
    country_stats = multipliers_df.groupby("country").agg({
        "multiplier": ["count", "mean", "std", "min", "max"],
        "z_score": "mean"
    }).round(2)
    
    country_stats.columns = ["Event_Count", "Avg_Multiplier", "Std_Multiplier", "Min_Multiplier", "Max_Multiplier", "Avg_Z_Score"]
    print(country_stats)
    
    # Z-score와 배수의 상관관계
    correlation = multipliers_df["z_score"].corr(multipliers_df["multiplier"])
    print(f"\nCorrelation between Z-score and multiplier: {correlation:.3f}")
    
    return multipliers_df, country_stats

def visualize_multipliers(multipliers_df):
    """배수 시각화"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. 전체 배수 분포
    axes[0, 0].hist(multipliers_df["multiplier"], bins=15, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0, 0].axvline(multipliers_df["multiplier"].mean(), color='red', linestyle='--', label=f'Mean: {multipliers_df["multiplier"].mean():.2f}x')
    axes[0, 0].axvline(multipliers_df["multiplier"].median(), color='orange', linestyle='--', label=f'Median: {multipliers_df["multiplier"].median():.2f}x')
    axes[0, 0].set_xlabel("Demand Multiplier")
    axes[0, 0].set_ylabel("Frequency")
    axes[0, 0].set_title("Distribution of Historical Event Multipliers")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. 국가별 평균 배수
    country_avg = multipliers_df.groupby("country")["multiplier"].mean().sort_values(ascending=False)
    axes[0, 1].bar(country_avg.index, country_avg.values, color='lightcoral')
    axes[0, 1].set_xlabel("Country")
    axes[0, 1].set_ylabel("Average Multiplier")
    axes[0, 1].set_title("Average Multiplier by Country")
    axes[0, 1].tick_params(axis='x', rotation=45)
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Z-score vs Multiplier
    axes[1, 0].scatter(multipliers_df["z_score"], multipliers_df["multiplier"], alpha=0.6, color='green')
    axes[1, 0].set_xlabel("Z-Score")
    axes[1, 0].set_ylabel("Multiplier")
    axes[1, 0].set_title("Z-Score vs Multiplier Relationship")
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. 시간별 배수 변화
    multipliers_df["year"] = multipliers_df["event_month"].dt.year
    year_avg = multipliers_df.groupby("year")["multiplier"].mean()
    axes[1, 1].plot(year_avg.index, year_avg.values, marker='o', linewidth=2, markersize=8)
    axes[1, 1].set_xlabel("Year")
    axes[1, 1].set_ylabel("Average Multiplier")
    axes[1, 1].set_title("Average Multiplier by Year")
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def suggest_multipliers(multipliers_df, country_stats):
    """배수 제안"""
    
    print("\n=== Suggested Multipliers Based on Historical Data ===")
    
    # 전체 평균 배수
    overall_avg = multipliers_df["multiplier"].mean()
    overall_median = multipliers_df["multiplier"].median()
    
    print(f"Overall average multiplier: {overall_avg:.2f}x")
    print(f"Overall median multiplier: {overall_median:.2f}x")
    
    # 국가별 제안
    print("\nCountry-specific suggestions:")
    for country in country_stats.index:
        avg_mult = country_stats.loc[country, "Avg_Multiplier"]
        event_count = country_stats.loc[country, "Event_Count"]
        
        if event_count >= 2:  # 충분한 데이터가 있는 국가
            print(f"  {country}: {avg_mult:.2f}x (based on {event_count} events)")
        else:
            print(f"  {country}: {overall_avg:.2f}x (using overall average, only {event_count} event)")
    
    # 확률 기반 배수 제안
    print("\n=== Probability-based Multiplier Suggestions ===")
    print("Based on Z-score percentiles:")
    
    z_scores = multipliers_df["z_score"].sort_values()
    percentiles = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
    
    for p in percentiles:
        idx = int(len(z_scores) * p)
        if idx < len(z_scores):
            z_threshold = z_scores.iloc[idx]
            corresponding_multipliers = multipliers_df[multipliers_df["z_score"] >= z_threshold]["multiplier"]
            avg_mult = corresponding_multipliers.mean()
            print(f"  Z-score >= {z_threshold:.2f} (top {100-p*100:.0f}%): {avg_mult:.2f}x")
    
    return overall_avg, overall_median

def main():
    """메인 실행"""
    print("Loading historical data...")
    demand_df = load_historical_data()
    
    print("Detecting historical events...")
    event_df = detect_historical_events(demand_df)
    
    print("Calculating event multipliers...")
    multipliers_df = calculate_event_multipliers(demand_df, event_df)
    
    if len(multipliers_df) == 0:
        print("No events detected! Try lowering the threshold.")
        return
    
    # 분석
    multipliers_df, country_stats = analyze_multiplier_patterns(multipliers_df)
    
    # 시각화
    print("\nGenerating visualizations...")
    visualize_multipliers(multipliers_df)
    
    # 배수 제안
    overall_avg, overall_median = suggest_multipliers(multipliers_df, country_stats)
    
    # 결과 저장
    output_path = DATA_DIR / "historical_multipliers.csv"
    multipliers_df.to_csv(output_path, index=False)
    print(f"\nDetailed results saved to: {output_path}")
    
    return multipliers_df, country_stats, overall_avg, overall_median

if __name__ == "__main__":
    results = main() 