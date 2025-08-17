"""
마케팅 지출과 수요 급등의 상관관계 분석
출제자의 "마케팅 지출 = 이벤트" 가정을 검증하는 코드
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# matplotlib 백엔드 설정 (GUI 없이 실행)
plt.switch_backend('Agg')

# 데이터 경로
DATA_DIR = Path("C:/projects/smartphone-supplychain/data")

def load_data():
    """모든 데이터 로드"""
    print("📊 데이터 로딩 중...")
    
    # 수요 데이터 (SQLite 데이터베이스)
    import sqlite3
    conn = sqlite3.connect(DATA_DIR / "demand_train.db")
    demand = pd.read_sql_query("SELECT * FROM demand_train", conn)
    conn.close()
    demand['date'] = pd.to_datetime(demand['date'])
    
    # 마케팅 지출 데이터
    marketing = pd.read_csv(DATA_DIR / "marketing_spend.csv", parse_dates=["date"])
    
    # 국가 매핑
    country_map = {
        'Seoul': 'KOR', 'Busan': 'KOR', 'Incheon': 'KOR', 'Daegu': 'KOR', 'Daejeon': 'KOR',
        'Gwangju': 'KOR', 'Suwon': 'KOR', 'Ulsan': 'KOR', 'Seongnam': 'KOR', 'Bucheon': 'KOR',
        'Tokyo': 'JPN', 'Yokohama': 'JPN', 'Osaka': 'JPN', 'Nagoya': 'JPN', 'Sapporo': 'JPN',
        'Fukuoka': 'JPN', 'Kobe': 'JPN', 'Kyoto': 'JPN', 'Kawasaki': 'JPN', 'Saitama': 'JPN',
        'New York': 'USA', 'Los Angeles': 'USA', 'Chicago': 'USA', 'Houston': 'USA', 'Phoenix': 'USA',
        'Philadelphia': 'USA', 'San Antonio': 'USA', 'San Diego': 'USA', 'Dallas': 'USA', 'San Jose': 'USA',
        'London': 'GBR', 'Birmingham': 'GBR', 'Leeds': 'GBR', 'Glasgow': 'GBR', 'Sheffield': 'GBR',
        'Bradford': 'GBR', 'Edinburgh': 'GBR', 'Liverpool': 'GBR', 'Manchester': 'GBR', 'Bristol': 'GBR',
        'Berlin': 'DEU', 'Hamburg': 'DEU', 'Munich': 'DEU', 'Cologne': 'DEU', 'Frankfurt': 'DEU',
        'Stuttgart': 'DEU', 'Düsseldorf': 'DEU', 'Dortmund': 'DEU', 'Essen': 'DEU', 'Leipzig': 'DEU',
        'Paris': 'FRA', 'Marseille': 'FRA', 'Lyon': 'FRA', 'Toulouse': 'FRA', 'Nice': 'FRA',
        'Nantes': 'FRA', 'Strasbourg': 'FRA', 'Montpellier': 'FRA', 'Bordeaux': 'FRA', 'Lille': 'FRA',
        'Toronto': 'CAN', 'Montreal': 'CAN', 'Vancouver': 'CAN', 'Calgary': 'CAN', 'Edmonton': 'CAN',
        'Ottawa': 'CAN', 'Winnipeg': 'CAN', 'Quebec City': 'CAN', 'Hamilton': 'CAN', 'Kitchener': 'CAN',
        'Sydney': 'AUS', 'Melbourne': 'AUS', 'Brisbane': 'AUS', 'Perth': 'AUS', 'Adelaide': 'AUS',
        'Gold Coast': 'AUS', 'Newcastle': 'AUS', 'Canberra': 'AUS', 'Sunshine Coast': 'AUS', 'Wollongong': 'AUS',
        'São Paulo': 'BRA', 'Rio de Janeiro': 'BRA', 'Brasília': 'BRA', 'Salvador': 'BRA', 'Fortaleza': 'BRA',
        'Belo Horizonte': 'BRA', 'Manaus': 'BRA', 'Curitiba': 'BRA', 'Recife': 'BRA', 'Porto Alegre': 'BRA',
        'Johannesburg': 'ZAR', 'Cape Town': 'ZAR', 'Durban': 'ZAR', 'Pretoria': 'ZAR', 'Port Elizabeth': 'ZAR',
        'Bloemfontein': 'ZAR', 'East London': 'ZAR', 'Kimberley': 'ZAR', 'Nelspruit': 'ZAR', 'Polokwane': 'ZAR'
    }
    
    demand['country'] = demand['city'].map(country_map)
    
    print(f"✅ 수요 데이터: {len(demand):,}행")
    print(f"✅ 마케팅 데이터: {len(marketing):,}행")
    
    return demand, marketing

def analyze_marketing_demand_correlation(demand, marketing):
    """마케팅 지출과 수요의 상관관계 분석"""
    print("\n🔍 마케팅 지출과 수요 상관관계 분석 중...")
    
    # 1. 국가별 일별 마케팅 지출 집계
    marketing_daily = marketing.groupby(['date', 'country'])['spend_usd'].sum().reset_index()
    
    # 2. 국가별 일별 수요 집계
    demand_daily = demand.groupby(['date', 'country'])['demand'].sum().reset_index()
    
    # 3. 마케팅과 수요 데이터 병합
    merged_data = demand_daily.merge(marketing_daily, on=['date', 'country'], how='left')
    merged_data['spend_usd'] = merged_data['spend_usd'].fillna(0)
    
    print(f"✅ 병합된 데이터: {len(merged_data):,}행")
    
    # 4. 전체 상관관계
    correlation = merged_data['demand'].corr(merged_data['spend_usd'])
    print(f"📊 전체 상관계수: {correlation:.4f}")
    
    # 5. 국가별 상관관계
    country_correlations = []
    for country in merged_data['country'].unique():
        country_data = merged_data[merged_data['country'] == country]
        if len(country_data) > 10:  # 최소 10개 데이터 포인트
            corr = country_data['demand'].corr(country_data['spend_usd'])
            country_correlations.append({
                'country': country,
                'correlation': corr,
                'data_points': len(country_data)
            })
    
    country_corr_df = pd.DataFrame(country_correlations)
    country_corr_df = country_corr_df.sort_values('correlation', ascending=False)
    
    print("\n📈 국가별 상관계수 (상위 10개):")
    print(country_corr_df.head(10))
    
    return merged_data, country_corr_df

def analyze_marketing_periods_impact(demand, marketing):
    """마케팅 지출 기간과 수요 변화 분석"""
    print("\n🎯 마케팅 지출 기간과 수요 변화 분석 중...")
    
    # 1. 마케팅 지출 기간 찾기 (출제자 방식)
    marketing_daily = marketing.groupby(['date', 'country'])['spend_usd'].sum().reset_index()
    
    marketing_periods = []
    
    for country in marketing_daily['country'].unique():
        country_data = marketing_daily[marketing_daily['country'] == country].sort_values('date')
        
        # 마케팅 지출이 있는 날짜들
        active_days = country_data[country_data['spend_usd'] > 0]
        
        if len(active_days) == 0:
            continue
            
        # 연속된 기간으로 그룹화
        active_days['date_diff'] = active_days['date'].diff().dt.days
        active_days['period_id'] = (active_days['date_diff'] > 1).cumsum()
        
        # 각 기간별 분석
        periods = active_days.groupby('period_id').agg(
            start_date=('date', 'min'),
            end_date=('date', 'max'),
            total_spend=('spend_usd', 'sum'),
            duration_days=('date', lambda x: (x.max() - x.min()).days + 1)
        ).reset_index()
        
        for _, period in periods.iterrows():
            marketing_periods.append({
                'country': country,
                'start_date': period['start_date'],
                'end_date': period['end_date'],
                'total_spend': period['total_spend'],
                'duration_days': period['duration_days']
            })
    
    marketing_periods_df = pd.DataFrame(marketing_periods)
    
    # 2. 각 마케팅 기간 전후 수요 변화 분석
    period_impacts = []
    
    for _, period in marketing_periods_df.iterrows():
        country = period['country']
        start_date = period['start_date']
        end_date = period['end_date']
        
        # 해당 국가의 수요 데이터
        country_demand = demand[demand['country'] == country].copy()
        
        # 마케팅 기간 전후 30일 비교
        before_start = start_date - pd.Timedelta(days=30)
        after_end = end_date + pd.Timedelta(days=30)
        
        # 기간별 평균 수요
        before_demand = country_demand[
            (country_demand['date'] >= before_start) & 
            (country_demand['date'] < start_date)
        ]['demand'].mean()
        
        during_demand = country_demand[
            (country_demand['date'] >= start_date) & 
            (country_demand['date'] <= end_date)
        ]['demand'].mean()
        
        after_demand = country_demand[
            (country_demand['date'] > end_date) & 
            (country_demand['date'] <= after_end)
        ]['demand'].mean()
        
        # 변화율 계산
        during_change = (during_demand - before_demand) / before_demand if before_demand > 0 else 0
        after_change = (after_demand - before_demand) / before_demand if before_demand > 0 else 0
        
        period_impacts.append({
            'country': country,
            'start_date': start_date,
            'end_date': end_date,
            'total_spend': period['total_spend'],
            'duration_days': period['duration_days'],
            'before_demand': before_demand,
            'during_demand': during_demand,
            'after_demand': after_demand,
            'during_change_pct': during_change * 100,
            'after_change_pct': after_change * 100
        })
    
    impacts_df = pd.DataFrame(period_impacts)
    
    print(f"✅ 분석된 마케팅 기간: {len(impacts_df)}개")
    print(f"📊 마케팅 기간 중 평균 수요 변화: {impacts_df['during_change_pct'].mean():.2f}%")
    print(f"📊 마케팅 기간 후 평균 수요 변화: {impacts_df['after_change_pct'].mean():.2f}%")
    
    return marketing_periods_df, impacts_df

def analyze_demand_surges_without_marketing(demand, marketing):
    """마케팅 없이 수요 급등이 발생한 경우 분석"""
    print("\n🚨 마케팅 없이 수요 급등 발생 케이스 분석 중...")
    
    # 1. 국가별 월별 수요 집계
    demand['year_month'] = demand['date'].dt.to_period('M')
    monthly_demand = demand.groupby(['country', 'year_month'])['demand'].sum().reset_index()
    
    # 2. Z-score 기반 수요 급등 감지
    demand_surges = []
    
    for country in monthly_demand['country'].unique():
        country_data = monthly_demand[monthly_demand['country'] == country].copy()
        country_data = country_data.sort_values('year_month')
        
        # 12개월 롤링 평균과 표준편차
        country_data['demand_mean'] = country_data['demand'].rolling(window=12, min_periods=1).mean()
        country_data['demand_std'] = country_data['demand'].rolling(window=12, min_periods=1).std()
        country_data['z_score'] = (country_data['demand'] - country_data['demand_mean']) / country_data['demand_std']
        
        # Z-score > 2인 경우를 수요 급등으로 정의
        surges = country_data[country_data['z_score'] > 2]
        
        for _, surge in surges.iterrows():
            # 해당 월에 마케팅 지출이 있었는지 확인
            surge_start = surge['year_month'].start_time
            surge_end = surge['year_month'].end_time
            
            country_marketing = marketing[marketing['country'] == country]
            marketing_in_period = country_marketing[
                (country_marketing['date'] >= surge_start) & 
                (country_marketing['date'] <= surge_end) &
                (country_marketing['spend_usd'] > 0)
            ]
            
            demand_surges.append({
                'country': country,
                'year_month': surge['year_month'],
                'demand': surge['demand'],
                'z_score': surge['z_score'],
                'normal_demand': surge['demand_mean'],
                'multiplier': surge['demand'] / surge['demand_mean'],
                'had_marketing': len(marketing_in_period) > 0,
                'marketing_spend': marketing_in_period['spend_usd'].sum() if len(marketing_in_period) > 0 else 0
            })
    
    surges_df = pd.DataFrame(demand_surges)
    
    print(f"✅ 감지된 수요 급등: {len(surges_df)}개")
    print(f"📊 마케팅과 함께 발생한 급등: {len(surges_df[surges_df['had_marketing'] == True])}개")
    print(f"📊 마케팅 없이 발생한 급등: {len(surges_df[surges_df['had_marketing'] == False])}개")
    
    if len(surges_df) > 0:
        print(f"📊 마케팅과 함께 발생한 급등의 평균 배수: {surges_df[surges_df['had_marketing'] == True]['multiplier'].mean():.2f}")
        print(f"📊 마케팅 없이 발생한 급등의 평균 배수: {surges_df[surges_df['had_marketing'] == False]['multiplier'].mean():.2f}")
    
    return surges_df

def create_visualizations(merged_data, country_corr_df, impacts_df, surges_df):
    """시각화 생성"""
    print("\n📊 시각화 생성 중...")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('마케팅 지출과 수요 급등 상관관계 분석', fontsize=16, fontweight='bold')
    
    # 1. 전체 상관관계 산점도
    ax1 = axes[0, 0]
    ax1.scatter(merged_data['spend_usd'], merged_data['demand'], alpha=0.5, s=1)
    ax1.set_xlabel('마케팅 지출 (USD)')
    ax1.set_ylabel('수요')
    ax1.set_title('마케팅 지출 vs 수요 (전체)')
    ax1.grid(True, alpha=0.3)
    
    # 2. 국가별 상관계수
    ax2 = axes[0, 1]
    top_countries = country_corr_df.head(10)
    bars = ax2.barh(range(len(top_countries)), top_countries['correlation'])
    ax2.set_yticks(range(len(top_countries)))
    ax2.set_yticklabels(top_countries['country'])
    ax2.set_xlabel('상관계수')
    ax2.set_title('국가별 마케팅-수요 상관계수 (상위 10개)')
    ax2.grid(True, alpha=0.3)
    
    # 색상 구분 (양의 상관관계 vs 음의 상관관계)
    for i, bar in enumerate(bars):
        if top_countries.iloc[i]['correlation'] > 0:
            bar.set_color('green')
        else:
            bar.set_color('red')
    
    # 3. 마케팅 기간 중 수요 변화
    ax3 = axes[1, 0]
    ax3.hist(impacts_df['during_change_pct'], bins=20, alpha=0.7, color='blue')
    ax3.axvline(impacts_df['during_change_pct'].mean(), color='red', linestyle='--', 
                label=f'평균: {impacts_df["during_change_pct"].mean():.1f}%')
    ax3.set_xlabel('수요 변화율 (%)')
    ax3.set_ylabel('빈도')
    ax3.set_title('마케팅 기간 중 수요 변화 분포')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. 마케팅 유무에 따른 수요 급등 비교
    ax4 = axes[1, 1]
    if len(surges_df) > 0:
        marketing_surges = surges_df[surges_df['had_marketing'] == True]['multiplier']
        no_marketing_surges = surges_df[surges_df['had_marketing'] == False]['multiplier']
        
        if len(marketing_surges) > 0 and len(no_marketing_surges) > 0:
            ax4.boxplot([marketing_surges, no_marketing_surges], 
                       labels=['마케팅 있음', '마케팅 없음'])
            ax4.set_ylabel('수요 배수')
            ax4.set_title('마케팅 유무에 따른 수요 급등 강도 비교')
            ax4.grid(True, alpha=0.3)
        else:
            ax4.text(0.5, 0.5, '데이터 부족', ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('마케팅 유무에 따른 수요 급등 비교')
    else:
        ax4.text(0.5, 0.5, '수요 급등 데이터 없음', ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title('마케팅 유무에 따른 수요 급등 비교')
    
    plt.tight_layout()
    plt.savefig(DATA_DIR / 'marketing_demand_correlation_analysis.png', dpi=300, bbox_inches='tight')
    print(f"✅ 시각화 저장: {DATA_DIR / 'marketing_demand_correlation_analysis.png'}")
    plt.close()

def generate_summary_report(merged_data, country_corr_df, impacts_df, surges_df):
    """분석 결과 요약 리포트 생성"""
    print("\n📋 분석 결과 요약 리포트 생성 중...")
    
    # 전체 상관계수
    overall_corr = merged_data['demand'].corr(merged_data['spend_usd'])
    
    # 마케팅 기간 중 수요 변화 통계
    positive_impacts = impacts_df[impacts_df['during_change_pct'] > 0]
    negative_impacts = impacts_df[impacts_df['during_change_pct'] < 0]
    
    # 수요 급등 통계
    marketing_surges = surges_df[surges_df['had_marketing'] == True] if len(surges_df) > 0 else pd.DataFrame()
    no_marketing_surges = surges_df[surges_df['had_marketing'] == False] if len(surges_df) > 0 else pd.DataFrame()
    
    report = f"""
# 마케팅 지출과 수요 급등 상관관계 분석 리포트

## 📊 전체 상관관계
- **전체 상관계수**: {overall_corr:.4f}
- **해석**: {'강한 양의 상관관계' if overall_corr > 0.7 else '중간 양의 상관관계' if overall_corr > 0.3 else '약한 양의 상관관계' if overall_corr > 0.1 else '약한 음의 상관관계' if overall_corr > -0.1 else '중간 음의 상관관계' if overall_corr > -0.3 else '강한 음의 상관관계'}

## 🎯 마케팅 기간 중 수요 변화 분석
- **분석된 마케팅 기간**: {len(impacts_df)}개
- **평균 수요 변화**: {impacts_df['during_change_pct'].mean():.2f}%
- **수요 증가한 기간**: {len(positive_impacts)}개 ({len(positive_impacts)/len(impacts_df)*100:.1f}%)
- **수요 감소한 기간**: {len(negative_impacts)}개 ({len(negative_impacts)/len(impacts_df)*100:.1f}%)

## 🚨 수요 급등 분석
- **총 수요 급등**: {len(surges_df)}개
- **마케팅과 함께 발생**: {len(marketing_surges)}개 ({(len(marketing_surges)/len(surges_df)*100) if len(surges_df) > 0 else 0:.1f}%)
- **마케팅 없이 발생**: {len(no_marketing_surges)}개 ({(len(no_marketing_surges)/len(surges_df)*100) if len(surges_df) > 0 else 0:.1f}%)

## 📈 상관계수 상위 국가 (상위 5개)
"""
    
    for i, row in country_corr_df.head(5).iterrows():
        report += f"- **{row['country']}**: {row['correlation']:.4f}\n"
    
    report += f"""
## 🎯 결론 및 권장사항

### 출제자 가정 검증 결과
1. **마케팅 지출과 수요의 상관관계**: {overall_corr:.4f} ({'강함' if abs(overall_corr) > 0.5 else '중간' if abs(overall_corr) > 0.3 else '약함'})
2. **마케팅 기간 중 수요 변화**: {impacts_df['during_change_pct'].mean():.2f}% ({'긍정적' if impacts_df['during_change_pct'].mean() > 0 else '부정적'})
3. **수요 급등과 마케팅의 연관성**: {(len(marketing_surges)/len(surges_df)*100) if len(surges_df) > 0 else 0:.1f}% ({'높음' if len(marketing_surges)/len(surges_df) > 0.7 else '중간' if len(marketing_surges)/len(surges_df) > 0.5 else '낮음' if len(surges_df) > 0 else '데이터 없음'})

### 권장사항
"""
    
    if overall_corr > 0.3:
        report += "- ✅ 마케팅 지출을 이벤트 지표로 사용하는 것이 타당함\n"
    else:
        report += "- ⚠️ 마케팅 지출과 수요의 상관관계가 약함 - 다른 지표 고려 필요\n"
    
    if impacts_df['during_change_pct'].mean() > 0:
        report += "- ✅ 마케팅 기간 중 수요가 증가하는 경향이 있음\n"
    else:
        report += "- ⚠️ 마케팅 기간 중 수요가 감소하는 경향이 있음\n"
    
    if len(surges_df) > 0 and len(marketing_surges)/len(surges_df) > 0.5:
        report += "- ✅ 수요 급등의 대부분이 마케팅과 연관됨\n"
    else:
        report += "- ⚠️ 수요 급등의 상당 부분이 마케팅과 무관함 - 다른 요인 고려 필요\n"
    
    # 리포트 저장
    with open(DATA_DIR / 'marketing_demand_analysis_report.md', 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"✅ 분석 리포트 저장: {DATA_DIR / 'marketing_demand_analysis_report.md'}")
    
    return report

def main():
    """메인 분석 함수"""
    print("🚀 마케팅 지출과 수요 급등 상관관계 분석 시작")
    print("=" * 60)
    
    # 1. 데이터 로드
    demand, marketing = load_data()
    
    # 2. 상관관계 분석
    merged_data, country_corr_df = analyze_marketing_demand_correlation(demand, marketing)
    
    # 3. 마케팅 기간 영향 분석
    marketing_periods_df, impacts_df = analyze_marketing_periods_impact(demand, marketing)
    
    # 4. 수요 급등 분석
    surges_df = analyze_demand_surges_without_marketing(demand, marketing)
    
    # 5. 시각화
    create_visualizations(merged_data, country_corr_df, impacts_df, surges_df)
    
    # 6. 리포트 생성
    report = generate_summary_report(merged_data, country_corr_df, impacts_df, surges_df)
    
    print("\n" + "=" * 60)
    print("✅ 마케팅 지출과 수요 급등 상관관계 분석 완료!")
    print("\n📋 주요 결과:")
    print(f"   - 전체 상관계수: {merged_data['demand'].corr(merged_data['spend_usd']):.4f}")
    print(f"   - 마케팅 기간 중 평균 수요 변화: {impacts_df['during_change_pct'].mean():.2f}%")
    if len(surges_df) > 0:
        print(f"   - 수요 급등 중 마케팅과 연관된 비율: {len(surges_df[surges_df['had_marketing'] == True])/len(surges_df)*100:.1f}%")
    
    print(f"\n📁 생성된 파일:")
    print(f"   - 시각화: {DATA_DIR / 'marketing_demand_correlation_analysis.png'}")
    print(f"   - 리포트: {DATA_DIR / 'marketing_demand_analysis_report.md'}")

if __name__ == "__main__":
    main() 