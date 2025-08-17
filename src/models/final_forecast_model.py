# src/models/final_forecast_model.py
# LightGBM 기반 고급 시계열 예측 모델 - 출제자 접근법 반영

import pandas as pd
import numpy as np
import sqlite3
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
import platform
import warnings
warnings.filterwarnings('ignore')
import time
from datetime import datetime
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from tqdm import tqdm
import lightgbm as lgb

# 경로 설정
SCRIPT = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT.parents[2]
DATA_DIR = Path("C:/projects/smartphone-supplychain/data")

def print_progress(message, start_time=None):
    """진행상황과 시간을 출력하는 헬퍼 함수"""
    current_time = datetime.now().strftime("%H:%M:%S")
    if start_time:
        elapsed = time.time() - start_time
        print(f"[{current_time}] ⏱️ {elapsed:.1f}초 - {message}")
    else:
        print(f"[{current_time}] {message}")

def get_country_mapping():
    """도시-국가 매핑"""
    return {
        'Washington_DC': 'USA', 'New_York': 'USA', 'Chicago': 'USA', 'Dallas': 'USA',
        'Berlin': 'DEU', 'Munich': 'DEU', 'Frankfurt': 'DEU', 'Hamburg': 'DEU',
        'Paris': 'FRA', 'Lyon': 'FRA', 'Marseille': 'FRA', 'Toulouse': 'FRA',
        'Seoul': 'KOR', 'Busan': 'KOR', 'Incheon': 'KOR', 'Gwangju': 'KOR',
        'Tokyo': 'JPN', 'Osaka': 'JPN', 'Nagoya': 'JPN', 'Fukuoka': 'JPN',
        'Manchester': 'GBR', 'London': 'GBR', 'Birmingham': 'GBR', 'Glasgow': 'GBR',
        'Ottawa': 'CAN', 'Toronto': 'CAN', 'Vancouver': 'CAN', 'Montreal': 'CAN',
        'Canberra': 'AUS', 'Sydney': 'AUS', 'Melbourne': 'AUS', 'Brisbane': 'AUS',
        'Brasilia': 'BRA', 'Sao_Paulo': 'BRA', 'Rio_de_Janeiro': 'BRA', 'Salvador': 'BRA',
        'Pretoria': 'ZAF', 'Johannesburg': 'ZAF', 'Cape_Town': 'ZAF', 'Durban': 'ZAF'
    }

def get_hardcoded_event_periods():
    """
    Define hardcoded event periods based on analysis
    Returns: Dictionary with country-year as key and (start_date, end_date) as value
    """
    event_periods = {
        ('KOR', 2018): ('2018-02-15', '2018-03-19'),
        ('JPN', 2019): ('2019-01-19', '2019-03-03'),
        ('USA', 2020): ('2020-01-26', '2020-04-04'),
        ('USA', 2021): ('2021-03-13', '2021-06-06'),
        ('KOR', 2022): ('2022-02-05', '2022-04-12'),
        ('KOR', 2023): ('2023-06-01', '2023-08-31'),
        ('JPN', 2024): ('2024-02-01', '2024-04-30')
    }
    return event_periods

def load_enhanced_training_data():
    """LightGBM 모델용 고급 학습 데이터 로드"""
    print_progress("=== LightGBM 모델용 고급 데이터 로드 ===")
    start_time = time.time()
    
    # 데이터 로드
    print_progress("📊 데이터 로드 중...", start_time)
    conn = sqlite3.connect(DATA_DIR / "demand_train.db")
    demand = pd.read_sql_query("SELECT * FROM demand_train", conn)
    conn.close()
    demand['date'] = pd.to_datetime(demand['date'])

    oil = pd.read_csv(DATA_DIR / "oil_price_processed.csv", parse_dates=["date"])
    currency = pd.read_csv(DATA_DIR / "currency_processed.csv", parse_dates=["date"])
    consumer_conf = pd.read_csv(DATA_DIR / "consumer_confidence_processed.csv", parse_dates=["date"])
    marketing = pd.read_csv(DATA_DIR / "marketing_spend.csv", parse_dates=["date"])
    weather = pd.read_csv(DATA_DIR / "weather.csv", parse_dates=["date"])
    calendar = pd.read_csv(DATA_DIR / "calendar.csv", parse_dates=["date"])
    sku_meta = pd.read_csv(DATA_DIR / "sku_meta.csv", parse_dates=["launch_date"])
    ppt = pd.read_csv(DATA_DIR / "price_promo_train.csv", parse_dates=["date"])
    
    print_progress("🔧 피처 엔지니어링 중...", start_time)
    
    # 기본 피처
    country_map = get_country_mapping()
    demand["country"] = demand["city"].map(country_map)
    demand["month"] = demand["date"].dt.month
    demand["dayofyear"] = demand["date"].dt.dayofyear
    demand["weekday"] = demand["date"].dt.weekday
    
    # 외부 데이터 병합
    # calendar 데이터는 date와 country로 병합
    demand = demand.merge(calendar[["date", "country", "season"]], on=["date", "country"], how="left")
    demand = demand.merge(sku_meta[["sku", "family", "storage_gb", "launch_date"]], on="sku", how="left")
    demand["days_since_launch"] = (demand["date"] - demand["launch_date"]).dt.days.clip(lower=0)

    oil['pct_change'] = oil['brent_usd'].pct_change()
    oil['volatility_7d'] = oil['pct_change'].rolling(7).std()
    demand = demand.merge(oil[['date', 'brent_usd', 'pct_change', 'volatility_7d']], on='date', how='left')
    
    fx_cols = ['EUR=X', 'KRW=X', 'JPY=X', 'GBP=X', 'CAD=X', 'AUD=X', 'BRL=X', 'ZAR=X']
    demand = demand.merge(currency[['date'] + fx_cols], on='date', how='left')
    
    demand = demand.merge(consumer_conf[['date', 'country', 'confidence_index']], on=['date', 'country'], how='left')
    
    marketing_agg = marketing.groupby(['date', 'country'])['spend_usd'].sum().reset_index()
    demand = demand.merge(marketing_agg, on=['date', 'country'], how='left')
    
    demand = demand.merge(weather[['date', 'country', 'avg_temp', 'humidity']], on=['date', 'country'], how="left")
    demand = demand.merge(ppt[['date', 'sku', 'city', 'unit_price', 'discount_pct']], on=['date', 'sku', 'city'], how="left")
    
    # 이벤트 기간 설정 (하드코딩)
    print_progress("📅 하드코딩된 이벤트 기간 설정 중...")
    event_periods = get_hardcoded_event_periods()
    
    # 이벤트 기간을 DataFrame으로 변환
    events_list = []
    for (country, year), (start_date, end_date) in event_periods.items():
        events_list.append({
            'country': country,
            'year': year,
            'start_date': pd.to_datetime(start_date),
            'end_date': pd.to_datetime(end_date)
        })
    events_df = pd.DataFrame(events_list)
    
    print_progress(f"✅ 설정된 이벤트: {len(events_df)}개")
    for _, event in events_df.iterrows():
        print(f"  - {event['country']} ({event['year']}): {event['start_date'].strftime('%Y-%m-%d')} ~ {event['end_date'].strftime('%Y-%m-%d')}")
    
    # is_event 컬럼 생성
    demand['is_event'] = 0
    
    for _, event in events_df.iterrows():
        mask = (
            (demand['country'] == event['country']) & 
            (demand['date'] >= event['start_date']) & 
            (demand['date'] <= event['end_date'])
        )
        demand.loc[mask, 'is_event'] = 1
    
    event_count = demand['is_event'].sum()
    print_progress(f"✅ 이벤트 기간 데이터 포인트: {event_count:,}개")
    
    # 시계열 피처 생성
    print_progress("📈 시계열 피처 생성 중...", start_time)
        
    # 시계열 피처 생성
    for col in ['demand']:
        if col in demand.columns:
            # Lag 피처
            for lag in [1, 3, 7, 14]:
                demand[f'{col}_lag_{lag}'] = demand.groupby(['city', 'sku'])[col].shift(lag)
            
            # Rolling 평균 (transform 사용으로 인덱스 문제 해결)
            for window in [7, 14]:
                demand[f'{col}_rolling_mean_{window}'] = demand.groupby(['city', 'sku'])[col].transform(lambda x: x.rolling(window, min_periods=1).mean())
            
            # Rolling 표준편차 (transform 사용으로 인덱스 문제 해결)
            for window in [7, 14]:
                demand[f'{col}_rolling_std_{window}'] = demand.groupby(['city', 'sku'])[col].transform(lambda x: x.rolling(window, min_periods=1).std())
    
    # 계절성 및 카테고리 변수 인코딩
    print_progress("🔤 카테고리 변수 인코딩 중...", start_time)
    
    # Label Encoding
    categorical_cols = ['city', 'sku', 'country', 'family', 'season']
    label_encoders = {}
    
    for col in categorical_cols:
        if col in demand.columns:
            le = LabelEncoder()
            demand[f'{col}_encoded'] = le.fit_transform(demand[col].astype(str))
            label_encoders[col] = le
    
    # 할인율 정규화
    demand['discount_pct'] = demand['discount_pct'] / 100
    
    print("=== Before fillna ===")
    for col in ["demand","unit_price","discount_pct","spend_usd","brent_usd","confidence_index"]:
        if col in demand.columns:
            print(col, "nan:", demand[col].isna().sum(), "min:", demand[col].min() if demand[col].notna().any() else None)

    # 어떤 조합이 비었는지 샘플
    print(demand[demand["unit_price"].isna()].head(10)[["date","city","sku","unit_price"]])

    # NaN 처리
    demand = demand.fillna(0)
    
    # 디버깅: 주요 피처 통계량 출력
    print_progress("🔍 디버깅: 데이터 로드 후 주요 피처 통계량 확인 중...", start_time)
    debug_cols = ['demand', 'demand_ratio', 'unit_price', 'discount_pct', 'spend_usd', 'brent_usd', 'confidence_index']
    for col in debug_cols:
        if col in demand.columns:
            print(f"  - {col}: Mean={demand[col].mean():.2f}, Std={demand[col].std():.2f}, Min={demand[col].min():.2f}, Max={demand[col].max():.2f}, NaN={demand[col].isnull().sum()}")
    print("--------------------------------------------------")
    print(demand.head(20))
    print_progress(f"✅ 데이터 로드 완료: {demand.shape}", start_time)
    print_progress(f"📈 이벤트 감지: {len(events_df)}개", start_time)
    
    return demand, events_df, label_encoders

def augment_event_data(demand_data, events_df, augmentation_factor=3):
    """
    이벤트 데이터 증강: 유사한 패턴을 가진 가상 데이터 생성
    
    Args:
        demand_data: 원본 수요 데이터
        events_df: 이벤트 정보 DataFrame
        augmentation_factor: 증강 배수 (기본값: 3)
    
    Returns:
        augmented_data: 증강된 데이터
    """
    print_progress(f"🔄 이벤트 데이터 증강 시작 (배수: {augmentation_factor})")
    
    # 이벤트 기간 데이터 추출
    event_data = demand_data[demand_data['is_event'] == 1].copy()
    
    if len(event_data) == 0:
        print("⚠️ 증강할 이벤트 데이터가 없습니다.")
        return demand_data
    
    print(f"  - 원본 이벤트 데이터: {len(event_data):,}개")
    
    augmented_list = [demand_data.copy()]  # 원본 데이터 포함
    
    for i in range(augmentation_factor - 1):  # 추가로 (augmentation_factor - 1)번 증강
        # 이벤트 데이터 복사
        augmented_event = event_data.copy()
        
        # 1. 노이즈 추가 (수요 변동성 시뮬레이션)
        noise_factor = 0.1  # 10% 노이즈
        demand_noise = np.random.normal(0, noise_factor, len(augmented_event))
        augmented_event['demand'] = augmented_event['demand'] * (1 + demand_noise)
        augmented_event['demand'] = np.maximum(0, augmented_event['demand'])  # 음수 방지
        
        # 2. 시계열 피처 재계산 (수요는 원래 스케일 기준)
        for col in ['demand']:
            if col in augmented_event.columns:
                # Lag 피처
                for lag in [1, 3, 7, 14]:
                    col_name = f'{col}_lag_{lag}'
                    if col_name in augmented_event.columns:
                        augmented_event[col_name] = augmented_event.groupby(['city', 'sku'])[col].shift(lag)
                
                # Rolling 평균
                for window in [7, 14]:
                    col_name = f'{col}_rolling_mean_{window}'
                    if col_name in augmented_event.columns:
                        augmented_event[col_name] = augmented_event.groupby(['city', 'sku'])[col].transform(
                            lambda x: x.rolling(window, min_periods=1).mean()
                        )
                
                # Rolling 표준편차
                for window in [7, 14]:
                    col_name = f'{col}_rolling_std_{window}'
                    if col_name in augmented_event.columns:
                        augmented_event[col_name] = augmented_event.groupby(['city', 'sku'])[col].transform(
                            lambda x: x.rolling(window, min_periods=1).std()
                        )
        
        # 3. 외부 요인에 약간의 변동성 추가
        external_cols = ['confidence_index', 'spend_usd', 'avg_temp', 'humidity', 'brent_usd']
        for col in external_cols:
            if col in augmented_event.columns:
                # 5% 이내의 작은 변동성 추가
                external_noise = np.random.normal(0, 0.05, len(augmented_event))
                augmented_event[col] = augmented_event[col] * (1 + external_noise)
        
        # 4. 환율 데이터에 변동성 추가
        fx_cols = ['EUR=X', 'KRW=X', 'JPY=X', 'GBP=X', 'CAD=X', 'AUD=X', 'BRL=X', 'ZAR=X']
        for col in fx_cols:
            if col in augmented_event.columns:
                fx_noise = np.random.normal(0, 0.02, len(augmented_event))  # 2% 변동성
                augmented_event[col] = augmented_event[col] * (1 + fx_noise)
        
        # 6. NaN 값 처리
        augmented_event = augmented_event.fillna(0)
        
        # 증강된 데이터를 리스트에 추가
        augmented_list.append(augmented_event)
        
        print(f"  - 증강 {i+1} 완료: {len(augmented_event):,}개")
    
    # 모든 데이터 합치기
    final_augmented = pd.concat(augmented_list, ignore_index=True)
    
    # 증강된 데이터에 고유 식별자 추가 (중복 제거 방지)
    final_augmented['augmentation_id'] = range(len(final_augmented))
    
    # 중복 제거는 하지 않음 (증강된 데이터는 같은 날짜-도시-SKU라도 다른 값)
    # final_augmented = final_augmented.drop_duplicates(
    #     subset=['date', 'city', 'sku'], 
    #     keep='first'
    # )
    
    print(f"  - 최종 증강 결과: {len(final_augmented):,}개 (원본: {len(demand_data):,}개)")
    print(f"  - 이벤트 데이터 비율: {final_augmented['is_event'].sum() / len(final_augmented) * 100:.2f}%")
    
    return final_augmented

def prepare_lightgbm_features(demand_data):
    """LightGBM 모델용 피처 준비"""
    print_progress("🔧 LightGBM 모델용 피처 준비 중...")
    start_time = time.time()
    
    # 기본 피처
    feature_cols = [
        # 시간 기반
        'month', 'dayofyear', 'weekday',
        # 제품 특성
        'storage_gb', 'days_since_launch',
        # 카테고리 (인코딩된 것)
        'city_encoded', 'sku_encoded', 'country_encoded', 'family_encoded', 'season_encoded',
        # 가격 및 할인
        'unit_price', 'discount_pct',
        # 날씨
        'avg_temp', 'humidity',
        # 외부 요인
        'brent_usd', 'pct_change', 'volatility_7d',
        'confidence_index', 'spend_usd',
        # 환율
        'EUR=X', 'KRW=X', 'JPY=X', 'GBP=X', 'CAD=X', 'AUD=X', 'BRL=X', 'ZAR=X',
        # 이벤트
        'is_event'
    ]
    
    # 시계열 피처 추가
    ts_features = [col for col in demand_data.columns if any(x in col for x in ['lag_', 'rolling_mean_', 'rolling_std_'])]
    feature_cols.extend(ts_features)
    
    # 실제 존재하는 컬럼만 필터링
    feature_cols = [col for col in feature_cols if col in demand_data.columns]

    # 사용자 요청: 특정 시계열 피처 제외 (demand 관련 피처는 유지)
    remove_features = [
        # 다중공선성 분석 결과에 따른 추가 제거 피처
        'dayofyear',  # month와 거의 완벽한 상관관계 (r=0.9965)
        # 추가 다중공선성 해결을 위한 피처 제거
    ]
    before_cnt = len(feature_cols)
    feature_cols = [col for col in feature_cols if col not in remove_features]
    removed_cnt = before_cnt - len(feature_cols)
    if removed_cnt > 0:
        print_progress(f"🧹 제외한 피처 수: {removed_cnt}개 ({[f for f in remove_features if f in demand_data.columns]})", start_time)
    
    print_progress(f"✅ 피처 준비 완료:", start_time)
    print_progress(f"  - 총 피처 수: {len(feature_cols)}개", start_time)
    print_progress(f"  - 시계열 피처: {len(ts_features)}개", start_time)
    
    return demand_data, feature_cols

def calculate_vif(X, feature_names):
    """VIF(Variance Inflation Factor) 계산"""
    print_progress("🔍 VIF 분석 중...")
    
    vif_data = []
    for i, feature in enumerate(feature_names):
        # 해당 피처를 제외한 나머지 피처들로 회귀
        X_temp = X.drop(columns=[feature])
        y_temp = X[feature]
        
        # 선형 회귀 모델 학습
        model = LinearRegression()
        model.fit(X_temp, y_temp)
        
        # R² 계산
        r_squared = model.score(X_temp, y_temp)
        
        # VIF 계산 (R²가 1에 가까우면 VIF가 매우 커짐)
        if r_squared < 0.999:  # 수치적 안정성을 위한 임계값
            vif = 1 / (1 - r_squared)
        else:
            vif = float('inf')
        
        vif_data.append({
            'feature': feature,
            'vif': vif,
            'r_squared': r_squared
        })
    
    vif_df = pd.DataFrame(vif_data)
    vif_df = vif_df.sort_values('vif', ascending=False)
    
    return vif_df

def analyze_multicollinearity(X, feature_names, start_time):
    """다중공선성 분석 (VIF + 상관관계)"""
    print_progress("🔍 다중공선성 분석 시작...", start_time)
    
    # 1. VIF 분석
    vif_df = calculate_vif(X, feature_names)
    
    # VIF 결과 출력
    print_progress("📊 VIF 분석 결과:", start_time)
    print("  - VIF > 10: 심각한 다중공선성")
    print("  - VIF > 5: 주의가 필요한 다중공선성")
    print("  - VIF > 2: 약간의 다중공선성")
    print()
    
    high_vif_features = vif_df[vif_df['vif'] > 10]
    moderate_vif_features = vif_df[(vif_df['vif'] > 5) & (vif_df['vif'] <= 10)]
    
    if not high_vif_features.empty:
        print("🚨 심각한 다중공선성 (VIF > 10):")
        for _, row in high_vif_features.iterrows():
            print(f"  - {row['feature']}: VIF={row['vif']:.2f}, R²={row['r_squared']:.4f}")
        print()
    
    if not moderate_vif_features.empty:
        print("⚠️ 주의가 필요한 다중공선성 (VIF > 5):")
        for _, row in moderate_vif_features.iterrows():
            print(f"  - {row['feature']}: VIF={row['vif']:.2f}, R²={row['r_squared']:.4f}")
        print()
    
    # VIF 결과 저장
    vif_csv_path = DATA_DIR / 'vif_analysis.csv'
    vif_df.to_csv(vif_csv_path, index=False)
    print_progress(f"📁 VIF 분석 결과 저장: {vif_csv_path}", start_time)
    
    # 2. 상관관계 분석
    print_progress("📊 상관관계 분석 중...", start_time)
    
    # 상관계수 계산
    corr_matrix = X.corr()
    
    # 높은 상관관계 찾기 (절댓값 > 0.8)
    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            corr_val = corr_matrix.iloc[i, j]
            if abs(corr_val) > 0.8:
                high_corr_pairs.append({
                    'feature1': corr_matrix.columns[i],
                    'feature2': corr_matrix.columns[j],
                    'correlation': corr_val
                })
    
    if high_corr_pairs:
        print("🔗 높은 상관관계 (|r| > 0.8):")
        high_corr_pairs.sort(key=lambda x: abs(x['correlation']), reverse=True)
        for pair in high_corr_pairs[:20]:  # 상위 20개만 출력
            print(f"  - {pair['feature1']} ↔ {pair['feature2']}: r={pair['correlation']:.4f}")
        print()
    
    # 상관관계 히트맵 저장
    plt.figure(figsize=(20, 16))
    
    # 상관계수 절댓값이 0.5 이상인 것만 표시
    mask = np.abs(corr_matrix) < 0.5
    sns.heatmap(corr_matrix, mask=mask, annot=False, cmap='RdBu_r', center=0,
                square=True, linewidths=0.5, cbar_kws={"shrink": .8})
    plt.title('Feature Correlation Matrix (|r| >= 0.5)', fontsize=16)
    plt.tight_layout()
    
    corr_png_path = DATA_DIR / 'feature_correlation_heatmap.png'
    plt.savefig(corr_png_path, dpi=300, bbox_inches='tight')
    plt.close()
    print_progress(f"📁 상관관계 히트맵 저장: {corr_png_path}", start_time)
    
    # 3. 피처 그룹별 분석
    print_progress("📊 피처 그룹별 다중공선성 분석...", start_time)
    
    # 시계열 피처 그룹
    lag_features = [f for f in feature_names if 'lag_' in f]
    rolling_mean_features = [f for f in feature_names if 'rolling_mean_' in f]
    rolling_std_features = [f for f in feature_names if 'rolling_std_' in f]
    
    print(f"  - Lag 피처 수: {len(lag_features)}")
    print(f"  - Rolling Mean 피처 수: {len(rolling_mean_features)}")
    print(f"  - Rolling Std 피처 수: {len(rolling_std_features)}")
    
    # 각 그룹 내에서 높은 VIF를 가진 피처들
    for group_name, group_features in [('Lag', lag_features), 
                                      ('Rolling Mean', rolling_mean_features),
                                      ('Rolling Std', rolling_std_features)]:
        if group_features:
            group_vif = vif_df[vif_df['feature'].isin(group_features)]
            high_vif_in_group = group_vif[group_vif['vif'] > 5]
            if not high_vif_in_group.empty:
                print(f"  - {group_name} 그룹 내 높은 VIF 피처:")
                for _, row in high_vif_in_group.iterrows():
                    print(f"    * {row['feature']}: VIF={row['vif']:.2f}")
    
    return vif_df, high_corr_pairs

def train_lightgbm_model(train_data, val_data, feature_cols, start_time):
    """LightGBM 모델 학습"""
    print_progress("🚀 LightGBM 모델 학습 시작...", start_time)
        
    # 데이터 준비
    X_train = train_data[feature_cols]
    y_train = train_data['demand']  # 타겟을 원래 수요로 변경
    X_val = val_data[feature_cols]
    y_val = val_data['demand']  # 타겟을 원래 수요로 변경
    
    print_progress(f"📊 학습 데이터: {X_train.shape}, 검증 데이터: {X_val.shape}", start_time)
    
    # 디버깅: 이벤트 데이터 분석
    print_progress("🔍 디버깅: 이벤트 데이터 분석 중...", start_time)
    train_event_count = train_data['is_event'].sum()
    train_total_count = len(train_data)
    val_event_count = val_data['is_event'].sum()
    val_total_count = len(val_data)
    
    print(f"  - 훈련 데이터: 총 {train_total_count:,}개 중 이벤트 {train_event_count:,}개 ({train_event_count/train_total_count*100:.2f}%)")
    print(f"  - 검증 데이터: 총 {val_total_count:,}개 중 이벤트 {val_event_count:,}개 ({val_event_count/val_total_count*100:.2f}%)")
    
    # 이벤트 기간의 수요 통계
    event_demand = train_data[train_data['is_event'] == 1]['demand']
    non_event_demand = train_data[train_data['is_event'] == 0]['demand']
    print(f"  - 이벤트 기간 수요: 평균={event_demand.mean():.1f}, 중앙값={event_demand.median():.1f}, 최대={event_demand.max():.1f}")
    print(f"  - 비이벤트 기간 수요: 평균={non_event_demand.mean():.1f}, 중앙값={non_event_demand.median():.1f}, 최대={non_event_demand.max():.1f}")
    print(f"  - 이벤트/비이벤트 수요 비율: {event_demand.mean()/non_event_demand.mean():.2f}배")
    print("--------------------------------------------------")
    
    # 디버깅: 학습/검증 데이터 통계량 출력
    print_progress("🔍 디버깅: 학습/검증 데이터 통계량 확인 중...", start_time)
    print(f"  - X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"  - X_val shape: {X_val.shape}, y_val shape: {y_val.shape}")
    print(f"  - y_train (demand) stats: Mean={y_train.mean():.4f}, Std={y_train.std():.4f}, Min={y_train.min():.4f}, Max={y_train.max():.4f}")
    print(f"  - y_val (demand) stats: Mean={y_val.mean():.4f}, Std={y_val.std():.4f}, Min={y_val.min():.4f}, Max={y_val.max():.4f}")
    print("--------------------------------------------------")
    
    # LightGBM 데이터셋 생성
    # 샘플 가중치 생성 (이벤트 기간에 더 높은 가중치)
    # is_event가 1인 경우 가중치 200 (대폭 증가), 0인 경우 가중치 1
    event_weight = 100
    weights = np.where(train_data['is_event'] == 1, event_weight, 1)
    
    # 추가 가중치 제거: event_intensity 미사용
    final_weights = weights
    
    print(f"  - 이벤트 가중치: {event_weight}")
    print(f"  - 이벤트 기간 평균 가중치: {final_weights[train_data['is_event'] == 1].mean():.1f}")
    print(f"  - 비이벤트 기간 평균 가중치: {final_weights[train_data['is_event'] == 0].mean():.1f}")
    print(f"  - 가중치 비율: {final_weights[train_data['is_event'] == 1].mean() / final_weights[train_data['is_event'] == 0].mean():.1f}배")
    # ----------------------------------

    # LightGBM 데이터셋 생성 시 weight 파라미터 추가
    train_dataset = lgb.Dataset(X_train, label=y_train, weight=final_weights)
    val_dataset = lgb.Dataset(X_val, label=y_val, reference=train_dataset)
    
    # LightGBM 파라미터 - 이벤트 예측에 최적화
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'num_leaves': 128,  # 증가: 더 복잡한 패턴 학습
        'learning_rate': 0.005,  # 감소: 더 세밀한 학습
        'feature_fraction': 0.8,  # 감소: 과적합 방지
        'bagging_fraction': 0.7,  # 감소: 과적합 방지
        'bagging_freq': 3,  # 증가: 더 자주 bagging
        'min_data_in_leaf': 20,  # 추가: 과적합 방지
        'min_gain_to_split': 0.1,  # 추가: 의미있는 분할만
        'verbose': -1,
        'random_state': 42
    }
    
    # 모델 학습
    print_progress("📚 모델 학습 중...", start_time)
    model = lgb.train(
        params,
        train_dataset,
        valid_sets=[train_dataset, val_dataset],
        valid_names=['train', 'valid'],
        num_boost_round=2000,
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=True),
            lgb.log_evaluation(period=100)
        ]
    )
    
    print_progress(f"✅ LightGBM 모델 학습 완료: {time.time() - start_time:.1f}초", start_time)

    # -----------------------------
    # 피처 중요도 계산 및 저장/시각화
    # -----------------------------
    try:
        print_progress("📊 피처 중요도 계산 중...", start_time)
        gain_importance = model.feature_importance(importance_type='gain')
        split_importance = model.feature_importance(importance_type='split')
        fi_df = pd.DataFrame({
            'feature': feature_cols,
            'gain': gain_importance,
            'split': split_importance
        })
        fi_df['gain_pct'] = fi_df['gain'] / (fi_df['gain'].sum() + 1e-9)
        fi_df['split_pct'] = fi_df['split'] / (fi_df['split'].sum() + 1e-9)
        fi_df = fi_df.sort_values('gain', ascending=False).reset_index(drop=True)

        # is_event 피처 중요도 확인
        is_event_importance = fi_df[fi_df['feature'] == 'is_event']
        if not is_event_importance.empty:
            print(f"🔍 is_event 피처 중요도: gain={is_event_importance['gain'].iloc[0]:.1f}, gain_pct={is_event_importance['gain_pct'].iloc[0]*100:.2f}%")
        else:
            print("⚠️ is_event 피처가 피처 중요도에 나타나지 않음")           


        # 저장
        fi_csv_path = DATA_DIR / 'lightgbm_feature_importance.csv'
        fi_df.to_csv(fi_csv_path, index=False)
        print_progress(f"📁 피처 중요도 CSV 저장: {fi_csv_path}", start_time)

        # 상위 30개 시각화
        top_n = min(30, len(fi_df))
        top_df = fi_df.head(top_n).iloc[::-1]  # 가독성을 위해 역순 표시
        plt.figure(figsize=(10, max(6, top_n * 0.3)))
        plt.barh(top_df['feature'], top_df['gain'])
        plt.title('LightGBM Feature Importance (gain 기준)')
        plt.xlabel('Gain')
        plt.tight_layout()
        fi_png_path = DATA_DIR / 'lightgbm_feature_importance_top30.png'
        plt.savefig(fi_png_path, dpi=300, bbox_inches='tight')
        plt.close()
        print_progress(f"📁 피처 중요도 그래프 저장: {fi_png_path}", start_time)

        # 콘솔 상위 20개 출력
        print("\n상위 중요 피처 (gain 기준) Top 20:")
        for i, row in fi_df.head(20).iterrows():
            print(f"  {i+1:2d}. {row['feature']}: gain={row['gain']:.1f}, split={row['split']}, gain_pct={row['gain_pct']*100:.2f}%")
        print()
    except Exception as e:
        print(f"⚠️ 피처 중요도 계산/저장 중 오류: {e}")

    return model

def create_lightgbm_validation_visualization(val_data, val_pred, start_time):
    """LightGBM 모델 2022년 검증 결과 시각화"""
    print_progress("📊 LightGBM 모델 2022년 검증 시각화 생성 중...", start_time)
    
    # val_pred를 val_data에 추가하여 인덱스 매칭 문제 해결
    val_data_with_pred = val_data.copy()
    val_data_with_pred['predicted_demand'] = val_pred
    
    # 디버깅: 샘플링된 데이터의 예측값 통계 확인
    print_progress("🔍 디버깅: 2022년 시각화 샘플 데이터 예측값 통계 확인 중...", start_time)
    sample_combinations = val_data_with_pred[['city', 'sku']].drop_duplicates().head(5)
    for i, (_, combo) in enumerate(sample_combinations.iterrows()):
        city, sku = combo['city'], combo['sku']
        combo_data = val_data_with_pred[(val_data_with_pred['city'] == city) & (val_data_with_pred['sku'] == sku)].copy()
        if not combo_data.empty:
            actual = combo_data['demand'].values
            predicted = combo_data['predicted_demand'].values
            print(f"  - {city}-{sku} (Actual): Mean={np.mean(actual):.2f}, Max={np.max(actual):.2f}")
            print(f"  - {city}-{sku} (Predicted): Mean={np.mean(predicted):.2f}, Max={np.max(predicted):.2f}, Non-zero ratio={np.sum(predicted > 0) / len(predicted) * 100:.2f}%")
    print("--------------------------------------------------")
    
    # 샘플링 (한국 4개 도시 고정)
    korean_cities = ['Seoul', 'Busan', 'Incheon', 'Gwangju']
    sample_combinations = []
    
    for city in korean_cities:
        city_data = val_data_with_pred[val_data_with_pred['city'] == city]
        if not city_data.empty:
            # 해당 도시의 첫 번째 SKU 선택
            first_sku = city_data['sku'].iloc[0]
            sample_combinations.append({'city': city, 'sku': first_sku})
    
    if len(sample_combinations) == 0:
        # 한국 도시가 없으면 기존 방식으로 fallback
        sample_combinations = val_data_with_pred[['city', 'sku']].drop_duplicates().head(5).to_dict('records')
    
    fig, axes = plt.subplots(len(sample_combinations), 1, figsize=(15, 4 * len(sample_combinations)))
    if len(sample_combinations) == 1:
        axes = [axes]
    
    for i, combo in enumerate(sample_combinations):
        city, sku = combo['city'], combo['sku']
        
        # 해당 city-sku 조합의 데이터 추출
        mask = (val_data_with_pred['city'] == city) & (val_data_with_pred['sku'] == sku)
        combo_data = val_data_with_pred[mask].sort_values('date')
        
        if len(combo_data) > 0:
            # 실제 수요 (원래 스케일)
            actual_vals = combo_data['demand'].values
            # 예측 수요 (원래 스케일)
            pred_vals = combo_data['predicted_demand'].values
            
            axes[i].plot(actual_vals, label='Actual', linewidth=2)
            axes[i].plot(pred_vals, label='Predicted', linewidth=2, linestyle='--')
            axes[i].set_title(f'2022 Validation - {city} - {sku}')
            axes[i].set_xlabel('Index')
            axes[i].set_ylabel('Demand')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
        else:
            axes[i].text(0.5, 0.5, 'No data', ha='center', va='center', transform=axes[i].transAxes)
            axes[i].set_title(f'2022 Validation - {city} - {sku}')
    
    plt.tight_layout()
    
    # 파일 저장
    output_path = DATA_DIR / "lightgbm_validation_2022.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print_progress(f"📁 Saved LightGBM 2022 validation plot: {output_path}", start_time)

def create_lightgbm_full_timeline_visualization(demand_data, result_df, start_time):
    """LightGBM 모델 2018-2024 전체 시계열 예측 시각화"""
    print_progress("📊 Creating LightGBM 2018-2024 full timeline plot...", start_time)
    
    # 디버깅: 전체 타임라인 시각화 데이터 예측값 통계 확인
    print_progress("🔍 Debug: Checking stats for full timeline visualization...", start_time)
    print(f"  - 2023-2024 예측값 (result_df): Mean={result_df['mean'].mean():.2f}, Max={result_df['mean'].max():.2f}, Non-zero ratio={np.sum(result_df['mean'] > 0) / len(result_df) * 100:.2f}%")
    print("--------------------------------------------------")
    
    # 실제 데이터 (2018-2022) - 로그 스케일에서 원래 스케일로 변환
    actual_data = demand_data[['date', 'city', 'sku', 'demand']].copy()
    actual_data['type'] = 'actual'
    
    # 예측 데이터 (2023-2024)
    pred_vis_data = result_df[['date', 'city', 'sku', 'mean']].copy()
    pred_vis_data = pred_vis_data.rename(columns={'mean': 'demand'})
    pred_vis_data['type'] = 'predicted'
    
    # 데이터 통합
    combined_data = pd.concat([actual_data[['date', 'city', 'sku', 'demand', 'type']], pred_vis_data], ignore_index=True)
    combined_data['date'] = pd.to_datetime(combined_data['date'])
    
    # 실제 데이터에서 존재하는 도시와 SKU 샘플링
    available_cities = combined_data['city'].unique()
    available_skus = combined_data['sku'].unique()
    
    # 상위 5개 도시와 3개 SKU 선택
    sample_cities = available_cities[:5] if len(available_cities) >= 5 else available_cities
    sample_skus = available_skus[:3] if len(available_skus) >= 3 else available_skus
    
    print(f"📊 Sample cities for plotting: {sample_cities}")
    print(f"📊 Sample SKUs for plotting: {sample_skus}")
    
    fig, axes = plt.subplots(len(sample_cities), len(sample_skus), figsize=(24, 18))
    fig.suptitle('LightGBM: 2018-2024 Full Timeline - Actual vs Predicted', fontsize=16, fontweight='bold')
    
    for i, city in enumerate(sample_cities):
        for j, sku in enumerate(sample_skus):
            city_sku_data = combined_data[(combined_data['city'] == city) & (combined_data['sku'] == sku)]
            
            if len(city_sku_data) > 0:
                city_sku_data = city_sku_data.sort_values('date')
                
                # 실제 데이터 (2018-2022)
                actual_mask = city_sku_data['type'] == 'actual'
                actual_plot = city_sku_data[actual_mask]
                
                # 예측 데이터 (2023-2024)
                pred_mask = city_sku_data['type'] == 'predicted'
                pred_plot = city_sku_data[pred_mask]
                
                if len(actual_plot) > 0:
                    axes[i, j].plot(actual_plot['date'], actual_plot['demand'], 
                                   label='Actual (2018-2022)', color='blue', linewidth=2)
                
                if len(pred_plot) > 0:
                    axes[i, j].plot(pred_plot['date'], pred_plot['demand'], 
                                   label='Predicted (2023-2024)', color='red', linewidth=2, linestyle='--')
                
                # 2023년 시작점 표시
                axes[i, j].axvline(x=pd.Timestamp('2023-01-01'), color='green', linestyle=':', 
                                   alpha=0.7, label='Forecast start')
                
                axes[i, j].set_title(f'{city} - {sku}', fontsize=12, fontweight='bold')
                axes[i, j].set_xlabel('Date')
                axes[i, j].set_ylabel('Demand')
                axes[i, j].legend()
                axes[i, j].grid(True, alpha=0.3)
                
                # x축 날짜 포맷팅
                axes[i, j].tick_params(axis='x', rotation=45)
                
                # y축 범위 설정
                if len(actual_plot) > 0 and len(pred_plot) > 0:
                    y_min = min(actual_plot['demand'].min(), pred_plot['demand'].min())
                    y_max = max(actual_plot['demand'].max(), pred_plot['demand'].max())
                    axes[i, j].set_ylim([y_min * 0.8, y_max * 1.2])
            else:
                axes[i, j].text(0.5, 0.5, '데이터 없음', ha='center', va='center', 
                               transform=axes[i, j].transAxes)
                axes[i, j].set_title(f'{city} - {sku}', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    
    # 파일 저장
    output_path = DATA_DIR / "lightgbm_full_timeline_2018_2024.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print_progress(f"📁 Saved LightGBM full timeline plot: {output_path}", start_time)

def predict_future_lightgbm(model, demand_data, feature_cols, label_encoders, start_time, events_df=None):
    """LightGBM 모델로 미래 예측"""
    print_progress("🔮 LightGBM 모델로 미래 예측 중...", start_time)
    
    # 2023-2024년 날짜 생성
    future_dates = pd.date_range(start='2023-01-01', end='2024-12-31', freq='D')

    # 2023-2024 외생/보조 피처 실제 값 로드 (평균값 사용 대신 실제 값 사용)
    oil = pd.read_csv(DATA_DIR / "oil_price_processed.csv", parse_dates=["date"])
    oil['pct_change'] = oil['brent_usd'].pct_change()
    oil['volatility_7d'] = oil['pct_change'].rolling(7).std()
    currency = pd.read_csv(DATA_DIR / "currency_processed.csv", parse_dates=["date"])
    consumer_conf = pd.read_csv(DATA_DIR / "consumer_confidence_processed.csv", parse_dates=["date"])
    marketing = pd.read_csv(DATA_DIR / "marketing_spend.csv", parse_dates=["date"])
    weather = pd.read_csv(DATA_DIR / "weather.csv", parse_dates=["date"])
    calendar = pd.read_csv(DATA_DIR / "calendar.csv", parse_dates=["date"])
    sku_meta = pd.read_csv(DATA_DIR / "sku_meta.csv", parse_dates=["launch_date"])

    fx_cols = ['EUR=X', 'KRW=X', 'JPY=X', 'GBP=X', 'CAD=X', 'AUD=X', 'BRL=X', 'ZAR=X']

    # 국가별(day, country) 공변량
    marketing_agg = marketing.groupby(['date', 'country'])['spend_usd'].sum().reset_index()
    country_cov = consumer_conf[['date', 'country', 'confidence_index']]
    country_cov = country_cov.merge(marketing_agg, on=['date', 'country'], how='left')
    country_cov = country_cov.merge(weather[['date', 'country', 'avg_temp', 'humidity']], on=['date', 'country'], how='left')
    country_cov = country_cov.merge(calendar[['date', 'country', 'season']], on=['date', 'country'], how='left')

    # 날짜(day) 공변량
    cov_date = oil[['date', 'brent_usd', 'pct_change', 'volatility_7d']]
    cov_date = cov_date.merge(currency[['date'] + fx_cols], on='date', how='left')

    # 최종 미래 공변량 테이블
    cov_future = country_cov.merge(cov_date, on='date', how='left')

    # 디버깅: cov 테이블 구조 미리보기
    try:
        print_progress("🔎 cov_future preview:", start_time)
        print(f"  - shape: {cov_future.shape}")
        print(f"  - columns: {list(cov_future.columns)}")
        print(cov_future.head(5))
    except Exception as e:
        print(f"[WARN] cov_future preview failed: {e}")
    
    # 2023-2024 이벤트 사전 준비 (하드코딩된 이벤트 기간 사용)
    event_periods = get_hardcoded_event_periods()
    print_progress("📢 2023-2024 확정 이벤트 구간:")
    for (country, year), (start_date, end_date) in event_periods.items():
        if year in [2023, 2024]:
            print(f"  - {country} | {start_date} ~ {end_date}")
    
    # 모든 city-sku 조합
    city_sku_combinations = demand_data[['city', 'sku']].drop_duplicates()
    
    result = []
    
    print_progress(f"📊 예측 대상: {len(city_sku_combinations)}개 조합 × {len(future_dates)}일 = {len(city_sku_combinations) * len(future_dates):,}개", start_time)
    
    # 각 city-sku 조합별로 예측
    for idx, (_, combo) in enumerate(tqdm(city_sku_combinations.iterrows(), total=len(city_sku_combinations), desc="예측 진행")):
        city, sku = combo['city'], combo['sku']
        
        # 해당 조합의 최근 데이터 (피처 생성용)
        recent_data = demand_data[(demand_data['city'] == city) & (demand_data['sku'] == sku)].tail(60)
        
        if len(recent_data) == 0:
            continue
        
        # demand 버퍼 (오토레그레시브 업데이트용)
        demand_buffer = recent_data['demand'].dropna().tolist()
        if len(demand_buffer) == 0:
            demand_buffer = [0.0]
        # 디버깅: 초기 버퍼 상태
        if idx < 3:
            print(f"[DBG] {city}-{sku} initial demand_buffer (last 5): {demand_buffer[-5:]}  size={len(demand_buffer)}")
        
        # 미래 데이터 생성
        for date in future_dates:
            # 기본 피처 생성
            future_row = {
                'date': date,
                'city': city,
                'sku': sku,
                'month': date.month,
                'dayofyear': date.dayofyear,
                'weekday': date.weekday(),
            }

            # 카테고리 변수 값 설정 (최근 데이터에서 가져오기)
            if len(recent_data) > 0:
                future_row['country'] = recent_data['country'].iloc[0]
                future_row['family'] = recent_data['family'].iloc[0]
                future_row['season'] = recent_data['season'].iloc[0]
            else:
                first_combo = demand_data[(demand_data['city'] == city) & (demand_data['sku'] == sku)]
                if len(first_combo) > 0:
                    future_row['country'] = first_combo['country'].iloc[0]
                    future_row['family'] = first_combo['family'].iloc[0]
                    future_row['season'] = first_combo['season'].iloc[0]
                else:
                    future_row['country'] = demand_data['country'].iloc[0]
                    future_row['family'] = demand_data['family'].iloc[0]
                    future_row['season'] = demand_data['season'].iloc[0]

            # 외생/보조 피처는 실제 2018~2024 테이블에서 당일 값을 조회하여 사용
            # country가 정해진 이후에만 가능
            lookup_country = future_row['country'] if 'country' in future_row else None
            # country/date 조인값
            cov_row = cov_future[(cov_future['date'] == date) & (cov_future['country'] == lookup_country)]
            if len(cov_row) > 0:
                cov_row = cov_row.iloc[0]
                future_row['confidence_index'] = cov_row.get('confidence_index', 0)
                future_row['spend_usd'] = cov_row.get('spend_usd', 0)
                future_row['avg_temp'] = cov_row.get('avg_temp', 0)
                future_row['humidity'] = cov_row.get('humidity', 0)
                future_row['season'] = cov_row.get('season', future_row.get('season', ''))
            else:
                # fallback: 0
                for col in ['confidence_index', 'spend_usd', 'avg_temp', 'humidity']:
                    future_row[col] = 0

            # 날짜 공변량
            cov_date_row = cov_date[cov_date['date'] == date]
            if len(cov_date_row) > 0:
                cov_date_row = cov_date_row.iloc[0]
                future_row['brent_usd'] = cov_date_row.get('brent_usd', 0)
                future_row['pct_change'] = cov_date_row.get('pct_change', 0)
                future_row['volatility_7d'] = cov_date_row.get('volatility_7d', 0)
                for fx in fx_cols:
                    future_row[fx] = cov_date_row.get(fx, 0)
            else:
                for col in ['brent_usd', 'pct_change', 'volatility_7d'] + fx_cols:
                    future_row[col] = 0

            # days_since_launch
            if sku in sku_meta['sku'].values:
                sku_launch_date = sku_meta[sku_meta['sku'] == sku]['launch_date'].iloc[0]
                if pd.notna(sku_launch_date):
                    future_row['days_since_launch'] = (date - sku_launch_date).days
                    future_row['days_since_launch'] = max(0, future_row['days_since_launch'])
                else:
                    print(f"sku {sku} has no launch date")
                    future_row['days_since_launch'] = 0
            else:
                future_row['days_since_launch'] = 0
                print(f"sku {sku} not found in sku_meta")

            # 카테고리 변수 인코딩
            for col in ['city', 'sku', 'country', 'family', 'season']:
                if col in label_encoders:
                    value_to_encode = str(future_row[col])
                    if value_to_encode not in label_encoders[col].classes_ and len(label_encoders[col].classes_) > 0:
                        value_to_encode = str(label_encoders[col].classes_[0])
                    future_row[f'{col}_encoded'] = label_encoders[col].transform([value_to_encode])[0]

            # is_event 설정 (하드코딩된 이벤트 기간 사용)
            event_periods = get_hardcoded_event_periods()
            is_event = 0
            for (country, year), (start_date, end_date) in event_periods.items():
                if (
                    country == future_row['country']
                    and future_row['date'] >= pd.to_datetime(start_date)
                    and future_row['date'] <= pd.to_datetime(end_date)
                ):
                    is_event = 1
                    break
            future_row['is_event'] = is_event

            # 수요(demand) 시계열 피처는 직전 값 버퍼에서 생성 (오토레그레시브)
            for lag in [1, 3, 7, 14]:
                if len(demand_buffer) >= lag:
                    future_row[f'demand_lag_{lag}'] = demand_buffer[-lag]
                else:
                    future_row[f'demand_lag_{lag}'] = demand_buffer[0] if len(demand_buffer) > 0 else 0
            for window in [7, 14]:
                if len(demand_buffer) > 0:
                    series = demand_buffer[-window:] if len(demand_buffer) >= window else demand_buffer
                    future_row[f'demand_rolling_mean_{window}'] = float(np.mean(series))
                    future_row[f'demand_rolling_std_{window}'] = float(np.std(series, ddof=1)) if len(series) > 1 else 0.0
                else:
                    future_row[f'demand_rolling_mean_{window}'] = 0.0
                    future_row[f'demand_rolling_std_{window}'] = 0.0

            # 할인/가격 기본 피처는 0 대신 최근값으로 채움 + 시계열 유지
            # 기본값 설정 (최근값)
            if 'unit_price' in recent_data.columns and len(recent_data) > 0:
                future_row['unit_price'] = float(recent_data['unit_price'].ffill().iloc[-1])
            if 'discount_pct' in recent_data.columns and len(recent_data) > 0:
                future_row['discount_pct'] = float(recent_data['discount_pct'].ffill().iloc[-1])

            for col in ['discount_pct', 'unit_price']:
                if col in recent_data.columns:
                    for lag in [1, 3, 7, 14]:
                        if len(recent_data) >= lag:
                            future_row[f'{col}_lag_{lag}'] = recent_data[col].iloc[-lag]
                        else:
                            future_row[f'{col}_lag_{lag}'] = recent_data[col].iloc[0] if len(recent_data) > 0 else 0
                    for window in [7, 14]:
                        vals = recent_data[col].tail(window)
                        future_row[f'{col}_rolling_mean_{window}'] = vals.mean() if len(vals) > 0 else (recent_data[col].mean() if len(recent_data) > 0 else 0)
                        # std는 제거된 경우가 많아 생성 생략

            # spend_usd는 2018~2024 실제 값 존재 → 당일 기준으로 실제값 기반 lag/rolling 생성
            if lookup_country is not None:
                for lag in [1, 3, 7, 14]:
                    lag_date = date - pd.Timedelta(days=lag)
                    v = cov_future.loc[(cov_future['date'] == lag_date) & (cov_future['country'] == lookup_country), 'spend_usd']
                    future_row[f'spend_usd_lag_{lag}'] = float(v.iloc[0]) if len(v) > 0 else 0.0
                for window in [7, 14]:
                    start_d = date - pd.Timedelta(days=window-1)
                    mask = (cov_future['country'] == lookup_country) & (cov_future['date'] >= start_d) & (cov_future['date'] <= date)
                    vals = cov_future.loc[mask, 'spend_usd']
                    future_row[f'spend_usd_rolling_mean_{window}'] = float(vals.mean()) if len(vals) > 0 else 0.0

            # 단일 행 예측 수행
            row_df = pd.DataFrame([future_row])
            # 누락된 피처 보정
            for col in feature_cols:
                if col not in row_df.columns:
                    row_df[col] = 0
            X_row = row_df[feature_cols]
            # 디버깅: 첫 2개 조합, 첫 일주일은 핵심 피처 로그
            if idx < 2 and date <= pd.Timestamp('2023-01-07'):
                check_cols = [c for c in feature_cols if (
                    c.startswith('demand_lag_') or c.startswith('demand_rolling_mean_') or c.startswith('demand_rolling_std_') or
                    c.startswith('unit_price') or c.startswith('discount_pct') or c.startswith('spend_usd')
                )]
                snap = X_row[check_cols].iloc[0]
                nz_ratio = (snap.replace(0, np.nan).notna().mean()) if len(snap) else 0
                print(f"[DBG] {city}-{sku} {date.date()} nz_ratio={nz_ratio:.2f} demand_lag_1={snap.get('demand_lag_1', np.nan)} spend_usd={snap.get('spend_usd', np.nan)} unit_price={snap.get('unit_price', np.nan)} discount_pct={snap.get('discount_pct', np.nan)}")
            pred_demand = float(model.predict(X_row)[0])
            
            # 예측값을 원래 스케일로 변환 후 정수로 변환 (음수 방지)
            pred_demand = int(max(0, round(pred_demand)))

            # 결과 저장 및 버퍼 업데이트
            future_row['mean'] = pred_demand
            result.append(future_row)
            # demand 버퍼에는 예측값 추가
            demand_buffer.append(pred_demand)
            if len(demand_buffer) > 60:
                demand_buffer = demand_buffer[-60:]
    
    # 결과 생성
    result_df = pd.DataFrame(result)[['sku', 'city', 'date', 'mean']]
    
    # 디버깅: 예측값 분포 확인
    print_progress(f"📊 예측값 통계 - 평균: {result_df['mean'].mean():.2f}, 중앙값: {result_df['mean'].median():.2f}, 최대: {result_df['mean'].max():.2f}, 최소: {result_df['mean'].min():.2f}", start_time)
    print_progress(f"📊 0이 아닌 예측값 비율: {(result_df['mean'] > 0).mean():.3f}", start_time)
    
    return result_df

def generate_lightgbm_forecast():
    """LightGBM 모델 기반 예측 생성"""
    print_progress("=== LightGBM 모델 기반 고급 예측 생성 ===")
    total_start_time = time.time()
    
    # 1. 데이터 로드
    demand_data, events_df, label_encoders = load_enhanced_training_data()

    # 2018~2024 확정 이벤트 구간 전체 프린트
    if events_df is not None and len(events_df) > 0:
        print("\n📢 확정된 이벤트 구간 목록 (2018~2024):")
        events_df_sorted = events_df.sort_values(['year','country','start_date']) if 'year' in events_df.columns else events_df
        for _, ev in events_df_sorted.iterrows():
            yr = ev['start_date'].year
            if 2018 <= yr <= 2024:
                print(f"  - {ev['country']} | {ev['start_date'].date()} ~ {ev['end_date'].date()} (year={yr})")
    
    # 2. 피처 준비
    demand_data, feature_cols = prepare_lightgbm_features(demand_data)
    
    # 3. 데이터 분할 (증강 전에 분할)
    print_progress("📊 데이터 분할 중...", total_start_time)
    train_data = demand_data[demand_data['date'] < '2022-01-01'].copy()
    val_data = demand_data[(demand_data['date'] >= '2022-01-01') & (demand_data['date'] < '2023-01-01')].copy()
    
    # 3.5. 훈련 데이터에만 증강 적용 (검증 데이터는 원본 유지)
    #print_progress("🔄 훈련 데이터 증강 단계 시작...", total_start_time)
    #train_data = augment_event_data(train_data, events_df, augmentation_factor=3)ㄴㄴ
    
    print_progress(f"📊 데이터 분할 완료 - Train: {len(train_data):,}개, Val: {len(val_data):,}개", total_start_time)
    
    # 4. 다중공선성 분석
    print_progress("🔍 다중공선성 분석 중...", total_start_time)
    X_train_sample = train_data[feature_cols].sample(n=min(10000, len(train_data)), random_state=42)  # 샘플링으로 속도 향상
    vif_df, high_corr_pairs = analyze_multicollinearity(X_train_sample, feature_cols, total_start_time)
    
    # 5. 모델 학습
    model = train_lightgbm_model(train_data, val_data, feature_cols, total_start_time)
    
    # 6. 검증 성능 평가
    print_progress("📈 검증 성능 평가 중...", total_start_time)
        
    X_val = val_data[feature_cols]
    y_val = val_data['demand']
    val_pred = model.predict(X_val)
    y_val_original = y_val
    
    # 성능 계산 (원래 스케일 기준)
    val_rmse = np.sqrt(mean_squared_error(y_val_original, val_pred))
    val_r2 = r2_score(y_val_original, val_pred)
    
    print_progress(f"📊 검증 성능 - RMSE: {val_rmse:.4f}, R²: {val_r2:.4f}", total_start_time)
    
    # 7. 2022년 검증 시각화
    print_progress("📊 2022년 검증 시각화 생성 중...", total_start_time)
    
    # 디버깅: 2022년 이벤트 기간 확인
    print_progress("🔍 디버깅: 2022년 이벤트 기간 확인 중...", total_start_time)
    val_events = val_data[val_data['is_event'] == 1]
    if not val_events.empty:
        print(f"  - 2022년 이벤트 기간: {len(val_events):,}개 데이터 포인트")
        print(f"  - 이벤트 기간 날짜 범위: {val_events['date'].min().date()} ~ {val_events['date'].max().date()}")
        print(f"  - 이벤트 기간 평균 수요: {val_events['demand'].mean():.1f}")
    else:
        print("  - ⚠️ 2022년에 이벤트 기간이 없음!")
    
    create_lightgbm_validation_visualization(val_data, val_pred, total_start_time)
    
    # 8. 미래 예측
    print_progress("🔮 미래 예측 중...", total_start_time)
    result_df = predict_future_lightgbm(model, demand_data, feature_cols, label_encoders, total_start_time, events_df=events_df)
    
    # 9. 결과 저장
    output_path = DATA_DIR / "lightgbm_forecast_submission.csv"
    result_df.to_csv(output_path, index=False)
    
    print_progress(f"✅ LightGBM 모델 예측 완료: {time.time() - total_start_time:.1f}초", total_start_time)
    print_progress(f"📁 결과 저장: {output_path}", total_start_time)
    print_progress(f"📊 총 예측 수: {len(result_df):,}", total_start_time)
    print_progress(f"📈 평균 수요: {result_df['mean'].mean():.1f}", total_start_time)
    print_progress(f"📊 검증 RMSE: {val_rmse:.4f}", total_start_time)
    
    # 10. 2018-2024 전체 시계열 예측 시각화
    print_progress("📊 2018-2024 전체 시계열 예측 시각화 생성 중...", total_start_time)
    create_lightgbm_full_timeline_visualization(demand_data, result_df, total_start_time)
    
    return result_df

def main():
    """메인 실행"""
    print_progress("=== LightGBM 모델 기반 고급 시계열 예측 ===")
    result_df = generate_lightgbm_forecast()
    print_progress("✅ LightGBM 모델 완료!")

if __name__ == "__main__":
    main() 