# src/analysis/comprehensive_eda.py
# 종합적인 EDA (Exploratory Data Analysis) - 출제자 접근법 참고

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sqlite3
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
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

def load_all_data():
    """모든 데이터 로드"""
    print("=== 데이터 로드 ===")
    
    # 1. 수요 데이터
    conn = sqlite3.connect(DATA_DIR / "demand_train.db")
    demand = pd.read_sql("SELECT * FROM demand_train", conn, parse_dates=['date'])
    conn.close()
    
    # 국가 매핑 추가
    country_map = get_country_mapping()
    demand["country"] = demand["city"].map(country_map)
    
    # 2. 외부 데이터
    oil = pd.read_csv(DATA_DIR / "oil_price.csv", parse_dates=["date"])
    currency = pd.read_csv(DATA_DIR / "currency.csv", parse_dates=["Date"])
    currency = currency.rename(columns={"Date": "date"})
    consumer_conf = pd.read_csv(DATA_DIR / "consumer_confidence.csv", parse_dates=["month"])
    marketing = pd.read_csv(DATA_DIR / "marketing_spend.csv", parse_dates=["date"])
    weather = pd.read_csv(DATA_DIR / "weather.csv", parse_dates=["date"])
    calendar = pd.read_csv(DATA_DIR / "calendar.csv", parse_dates=["date"])
    sku_meta = pd.read_csv(DATA_DIR / "sku_meta.csv", parse_dates=["launch_date"])
    
    print(f"수요 데이터: {demand.shape}")
    print(f"유가 데이터: {oil.shape}")
    print(f"환율 데이터: {currency.shape}")
    print(f"소비자신뢰지수: {consumer_conf.shape}")
    print(f"마케팅 지출: {marketing.shape}")
    print(f"날씨 데이터: {weather.shape}")
    print(f"캘린더: {calendar.shape}")
    print(f"SKU 메타: {sku_meta.shape}")
    
    return demand, oil, currency, consumer_conf, marketing, weather, calendar, sku_meta

def analyze_demand_patterns(demand):
    """수요 패턴 분석"""
    print("\n=== 수요 패턴 분석 ===")
    
    # 1. 기본 통계
    print("1. 기본 통계:")
    print(f"   총 수요: {demand['demand'].sum():,}")
    print(f"   평균 수요: {demand['demand'].mean():.2f}")
    print(f"   중앙값: {demand['demand'].median():.2f}")
    print(f"   표준편차: {demand['demand'].std():.2f}")
    print(f"   최소값: {demand['demand'].min()}")
    print(f"   최대값: {demand['demand'].max()}")
    
    # 2. 시간별 패턴
    demand['year'] = demand['date'].dt.year
    demand['month'] = demand['date'].dt.month
    demand['weekday'] = demand['date'].dt.weekday
    demand['quarter'] = demand['date'].dt.quarter
    
    print("\n2. 연도별 수요:")
    yearly_demand = demand.groupby('year')['demand'].agg(['sum', 'mean', 'std']).round(2)
    print(yearly_demand)
    
    print("\n3. 월별 수요 패턴:")
    monthly_demand = demand.groupby('month')['demand'].mean().round(2)
    print(monthly_demand)
    
    print("\n4. 요일별 수요 패턴:")
    weekday_demand = demand.groupby('weekday')['demand'].mean().round(2)
    print(weekday_demand)
    
    # 3. 국가별 패턴
    print("\n5. 국가별 수요:")
    country_demand = demand.groupby('country')['demand'].agg(['sum', 'mean', 'std']).round(2)
    country_demand = country_demand.sort_values('sum', ascending=False)
    print(country_demand)
    
    # 4. SKU 패턴
    print("\n6. SKU별 수요:")
    sku_demand = demand.groupby('sku')['demand'].agg(['sum', 'mean', 'std']).round(2)
    sku_demand = sku_demand.sort_values('sum', ascending=False)
    print(sku_demand.head(10))
    
    return demand

def analyze_external_factors(oil, currency, consumer_conf, marketing, weather):
    """외부 요인 분석"""
    print("\n=== 외부 요인 분석 ===")
    
    # 1. 유가 분석
    print("1. 유가 분석:")
    print(f"   평균: ${oil['brent_usd'].mean():.2f}")
    print(f"   최소: ${oil['brent_usd'].min():.2f}")
    print(f"   최대: ${oil['brent_usd'].max():.2f}")
    print(f"   변동성: {oil['brent_usd'].std():.2f}")
    
    # 유가 변동성 계산
    oil['pct_change'] = oil['brent_usd'].pct_change()
    oil['volatility_7d'] = oil['pct_change'].rolling(7).std()
    
    print(f"   일일 변동성 평균: {oil['pct_change'].std():.3f}")
    print(f"   7일 변동성 평균: {oil['volatility_7d'].mean():.3f}")
    
    # 2. 환율 분석
    print("\n2. 환율 분석:")
    fx_cols = [col for col in currency.columns if col != 'date']
    for col in fx_cols:
        print(f"   {col}: 평균 {currency[col].mean():.4f}, 변동성 {currency[col].std():.4f}")
    
    # 3. 소비자신뢰지수 분석
    print("\n3. 소비자신뢰지수 분석:")
    country_conf = consumer_conf.groupby('country')['confidence_index'].agg(['mean', 'std']).round(2)
    print(country_conf)
    
    # 4. 마케팅 지출 분석
    print("\n4. 마케팅 지출 분석:")
    marketing_summary = marketing.groupby('country')['spend_usd'].agg(['sum', 'mean', 'std']).round(2)
    print(marketing_summary)
    
    # 5. 날씨 분석
    print("\n5. 날씨 분석:")
    weather_summary = weather.groupby('country')[['avg_temp', 'humidity']].agg(['mean', 'std']).round(2)
    print(weather_summary)
    
    return oil, currency, consumer_conf, marketing, weather

def detect_events_using_zscore(demand, threshold=2.0):
    """Z-score 기반 이벤트 탐지 (출제자 방식)"""
    print("\n=== Z-score 기반 이벤트 탐지 ===")
    
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
    
    events_df = pd.DataFrame(events_detected)
    
    if len(events_df) > 0:
        print(f"감지된 이벤트 수: {len(events_df)}")
        print("\n상위 이벤트들:")
        top_events = events_df.nlargest(10, 'z_score')[['country', 'date', 'z_score', 'multiplier']]
        print(top_events)
        
        # 국가별 이벤트 빈도
        print("\n국가별 이벤트 빈도:")
        country_freq = events_df['country'].value_counts()
        print(country_freq)
        
        # 월별 이벤트 분포
        print("\n월별 이벤트 분포:")
        events_df['month'] = events_df['date'].dt.month
        month_freq = events_df['month'].value_counts().sort_index()
        print(month_freq)
    else:
        print("임계값이 너무 높습니다. 낮춰서 다시 시도해보세요.")
    
    return events_df

def analyze_consumer_confidence_patterns(consumer_conf):
    """소비자신뢰지수 패턴 분석 (출제자 방식)"""
    print("\n=== 소비자신뢰지수 패턴 분석 ===")
    
    # 1. 캐나다 결측치 확인
    countries = consumer_conf['country'].unique()
    print(f"1. 포함된 국가: {list(countries)}")
    
    if 'CAN' not in countries:
        print("   ⚠️ 캐나다(CAN) 데이터 없음 - PCA로 추정 필요")
    
    # 2. 국가별 통계
    print("\n2. 국가별 통계:")
    country_stats = consumer_conf.groupby('country')['confidence_index'].agg(['mean', 'std', 'min', 'max']).round(2)
    print(country_stats)
    
    # 3. 시계열 패턴
    print("\n3. 시계열 패턴:")
    consumer_conf['year'] = consumer_conf['month'].dt.year
    yearly_pattern = consumer_conf.groupby(['country', 'year'])['confidence_index'].mean().unstack()
    print(yearly_pattern.round(2))
    
    # 4. 이상치 탐지
    print("\n4. 이상치 탐지 (Z-score > 3):")
    consumer_conf['z_score'] = consumer_conf.groupby('country')['confidence_index'].transform(
        lambda x: (x - x.mean()) / x.std()
    )
    outliers = consumer_conf[consumer_conf['z_score'].abs() > 3]
    print(f"   이상치 수: {len(outliers)}")
    if len(outliers) > 0:
        print(outliers[['country', 'month', 'confidence_index', 'z_score']].head())
    
    return consumer_conf

def create_global_confidence_factor(consumer_conf):
    """글로벌 신뢰지수 요인 생성 (출제자 방식)"""
    print("\n=== 글로벌 신뢰지수 요인 생성 ===")
    
    # 피벗 테이블 생성
    wide = consumer_conf.pivot(index="month", columns="country", values="confidence_index").sort_index()
    
    # 결측치 처리
    wide = wide.fillna(method='ffill')
    
    print(f"1. 피벗 테이블 형태: {wide.shape}")
    print(f"2. 포함된 국가: {list(wide.columns)}")
    
    # 표준화
    scaler = StandardScaler()
    Z = scaler.fit_transform(wide)
    
    # PCA로 글로벌 요인 추출
    pca = PCA(n_components=2)
    global_factors = pca.fit_transform(Z)
    
    print(f"3. PCA 설명 분산 비율:")
    print(f"   PC1: {pca.explained_variance_ratio_[0]:.3f}")
    print(f"   PC2: {pca.explained_variance_ratio_[1]:.3f}")
    print(f"   총 설명 분산: {pca.explained_variance_ratio_.sum():.3f}")
    
    # 글로벌 요인을 시계열로 변환
    global_factor_df = pd.DataFrame({
        'month': wide.index,
        'global_factor_1': global_factors[:, 0],
        'global_factor_2': global_factors[:, 1]
    })
    
    print(f"4. 글로벌 요인 생성 완료: {global_factor_df.shape}")
    
    return global_factor_df, pca

def analyze_price_promotion_patterns():
    """가격 및 프로모션 패턴 분석"""
    print("\n=== 가격 및 프로모션 패턴 분석 ===")
    
    # 가격 프로모션 데이터 로드
    ppt = pd.read_csv(DATA_DIR / "price_promo_train.csv", parse_dates=['date'])
    
    print(f"1. 데이터 크기: {ppt.shape}")
    print(f"2. 기간: {ppt['date'].min()} ~ {ppt['date'].max()}")
    
    # 기본 통계
    print("\n3. 가격 통계:")
    print(f"   평균 가격: ${ppt['unit_price'].mean():.2f}")
    print(f"   최소 가격: ${ppt['unit_price'].min():.2f}")
    print(f"   최대 가격: ${ppt['unit_price'].max():.2f}")
    
    print("\n4. 할인율 통계:")
    print(f"   평균 할인율: {ppt['discount_pct'].mean():.1%}")
    print(f"   최대 할인율: {ppt['discount_pct'].max():.1%}")
    print(f"   할인 없는 비율: {(ppt['discount_pct'] == 0).mean():.1%}")
    
    # SKU별 분석
    print("\n5. SKU별 평균 가격 (상위 10개):")
    sku_price = ppt.groupby('sku')['unit_price'].mean().sort_values(ascending=False)
    print(sku_price.head(10))
    
    # 도시별 분석
    print("\n6. 도시별 평균 가격:")
    city_price = ppt.groupby('city')['unit_price'].mean().sort_values(ascending=False)
    print(city_price.head(10))
    
    # 시간별 패턴
    ppt['year'] = ppt['date'].dt.year
    ppt['month'] = ppt['date'].dt.month
    
    print("\n7. 연도별 평균 가격:")
    yearly_price = ppt.groupby('year')['unit_price'].mean()
    print(yearly_price)
    
    print("\n8. 월별 평균 할인율:")
    monthly_discount = ppt.groupby('month')['discount_pct'].mean()
    print(monthly_discount)
    
    return ppt

def create_visualizations(demand, oil, events_df, consumer_conf, ppt):
    """종합 시각화"""
    print("\n=== 종합 시각화 생성 ===")
    
    fig, axes = plt.subplots(3, 3, figsize=(20, 15))
    
    # 1. 전체 수요 시계열
    daily_demand = demand.groupby('date')['demand'].sum()
    axes[0,0].plot(daily_demand.index, daily_demand.values, linewidth=1, alpha=0.8)
    axes[0,0].set_title('Daily Total Demand')
    axes[0,0].set_xlabel('Date')
    axes[0,0].set_ylabel('Demand')
    axes[0,0].grid(True, alpha=0.3)
    
    # 2. 국가별 수요 분포
    country_demand = demand.groupby('country')['demand'].sum().sort_values(ascending=True)
    axes[0,1].barh(range(len(country_demand)), country_demand.values)
    axes[0,1].set_yticks(range(len(country_demand)))
    axes[0,1].set_yticklabels(country_demand.index)
    axes[0,1].set_title('Total Demand by Country')
    axes[0,1].set_xlabel('Total Demand')
    
    # 3. 월별 수요 패턴
    monthly_demand = demand.groupby('month')['demand'].mean()
    axes[0,2].bar(monthly_demand.index, monthly_demand.values)
    axes[0,2].set_title('Average Demand by Month')
    axes[0,2].set_xlabel('Month')
    axes[0,2].set_ylabel('Average Demand')
    
    # 4. 유가 시계열
    axes[1,0].plot(oil['date'], oil['brent_usd'], linewidth=1, alpha=0.8)
    axes[1,0].set_title('Oil Price (Brent USD)')
    axes[1,0].set_xlabel('Date')
    axes[1,0].set_ylabel('Price (USD)')
    axes[1,0].grid(True, alpha=0.3)
    
    # 5. 이벤트 분포 (있는 경우)
    if len(events_df) > 0:
        event_counts = events_df['country'].value_counts()
        axes[1,1].bar(range(len(event_counts)), event_counts.values)
        axes[1,1].set_xticks(range(len(event_counts)))
        axes[1,1].set_xticklabels(event_counts.index, rotation=45)
        axes[1,1].set_title('Events by Country')
        axes[1,1].set_ylabel('Number of Events')
    
    # 6. 소비자신뢰지수 시계열
    for country in consumer_conf['country'].unique():
        subset = consumer_conf[consumer_conf['country'] == country]
        axes[1,2].plot(subset['month'], subset['confidence_index'], label=country, alpha=0.7)
    axes[1,2].set_title('Consumer Confidence Index')
    axes[1,2].set_xlabel('Month')
    axes[1,2].set_ylabel('Confidence Index')
    axes[1,2].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[1,2].grid(True, alpha=0.3)
    
    # 7. 가격 분포
    axes[2,0].hist(ppt['unit_price'], bins=50, alpha=0.7, edgecolor='black')
    axes[2,0].set_title('Price Distribution')
    axes[2,0].set_xlabel('Unit Price')
    axes[2,0].set_ylabel('Frequency')
    
    # 8. 할인율 분포
    axes[2,1].hist(ppt['discount_pct'], bins=30, alpha=0.7, edgecolor='black')
    axes[2,1].set_title('Discount Rate Distribution')
    axes[2,1].set_xlabel('Discount Rate')
    axes[2,1].set_ylabel('Frequency')
    
    # 9. 수요 vs 가격 산점도 (샘플)
    sample_data = ppt.sample(n=10000, random_state=42)
    axes[2,2].scatter(sample_data['unit_price'], sample_data['discount_pct'], alpha=0.5)
    axes[2,2].set_title('Price vs Discount Rate')
    axes[2,2].set_xlabel('Unit Price')
    axes[2,2].set_ylabel('Discount Rate')
    
    plt.tight_layout()
    plt.show()
    
    print("시각화 완료!")

def generate_insights(demand, events_df, consumer_conf, oil, ppt):
    """핵심 인사이트 생성"""
    print("\n=== 핵심 인사이트 ===")
    
    insights = []
    
    # 1. 수요 패턴 인사이트
    monthly_pattern = demand.groupby('month')['demand'].mean()
    peak_month = monthly_pattern.idxmax()
    low_month = monthly_pattern.idxmin()
    insights.append(f"수요는 {peak_month}월에 최고점, {low_month}월에 최저점")
    
    # 2. 국가별 인사이트
    country_demand = demand.groupby('country')['demand'].sum()
    top_country = country_demand.idxmax()
    insights.append(f"가장 높은 수요: {top_country}")
    
    # 3. 이벤트 인사이트
    if len(events_df) > 0:
        avg_multiplier = events_df['multiplier'].mean()
        insights.append(f"이벤트 시 평균 {avg_multiplier:.1f}배 수요 증가")
        
        most_event_country = events_df['country'].value_counts().index[0]
        insights.append(f"가장 많은 이벤트: {most_event_country}")
    
    # 4. 유가 인사이트
    oil_volatility = oil['brent_usd'].pct_change().std()
    insights.append(f"유가 변동성: {oil_volatility:.3f}")
    
    # 5. 가격 인사이트
    price_range = ppt['unit_price'].max() - ppt['unit_price'].min()
    insights.append(f"가격 범위: ${price_range:.2f}")
    
    # 6. 할인 인사이트
    discount_freq = (ppt['discount_pct'] > 0).mean()
    insights.append(f"할인 빈도: {discount_freq:.1%}")
    
    print("주요 발견사항:")
    for i, insight in enumerate(insights, 1):
        print(f"{i}. {insight}")
    
    return insights

def main():
    """메인 EDA 실행"""
    print("=== 종합 EDA 시작 ===\n")
    
    # 1. 데이터 로드
    demand, oil, currency, consumer_conf, marketing, weather, calendar, sku_meta = load_all_data()
    
    # 2. 수요 패턴 분석
    demand = analyze_demand_patterns(demand)
    
    # 3. 외부 요인 분석
    oil, currency, consumer_conf, marketing, weather = analyze_external_factors(
        oil, currency, consumer_conf, marketing, weather
    )
    
    # 4. 이벤트 탐지 (출제자 방식)
    events_df = detect_events_using_zscore(demand, threshold=2.0)
    
    # 5. 소비자신뢰지수 분석 (출제자 방식)
    consumer_conf = analyze_consumer_confidence_patterns(consumer_conf)
    
    # 6. 글로벌 요인 생성 (출제자 방식)
    global_factor_df, pca = create_global_confidence_factor(consumer_conf)
    
    # 7. 가격 프로모션 분석
    ppt = analyze_price_promotion_patterns()
    
    # 8. 시각화
    create_visualizations(demand, oil, events_df, consumer_conf, ppt)
    
    # 9. 핵심 인사이트 생성
    insights = generate_insights(demand, events_df, consumer_conf, oil, ppt)
    
    print("\n=== EDA 완료 ===")
    print("다음 단계: 발견된 패턴을 바탕으로 모델링 전략 수립")
    
    return {
        'demand': demand,
        'events_df': events_df,
        'consumer_conf': consumer_conf,
        'global_factor_df': global_factor_df,
        'oil': oil,
        'ppt': ppt,
        'insights': insights
    }

if __name__ == "__main__":
    results = main() 