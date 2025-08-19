# src/models/xgboost_model.py
# XGBoost 기반 고급 시계열 예측 모델 - final_forecast_model과 동일 로직, 모델만 XGBoost로 교체

import pandas as pd
import numpy as np
import sqlite3
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
import time
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from tqdm import tqdm
import xgboost as xgb

# 경로 설정 (final_forecast_model와 동일)
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
    """LightGBM 모델용 고급 학습 데이터 로드 (동일 로직 재현)"""
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
    for col in ['demand']:
        if col in demand.columns:
            for lag in [1, 3, 7, 14]:
                demand[f'{col}_lag_{lag}'] = demand.groupby(['city', 'sku'])[col].shift(lag)
            for window in [7, 14]:
                demand[f'{col}_rolling_mean_{window}'] = demand.groupby(['city', 'sku'])[col].transform(lambda x: x.rolling(window, min_periods=1).mean())
            for window in [7, 14]:
                demand[f'{col}_rolling_std_{window}'] = demand.groupby(['city', 'sku'])[col].transform(lambda x: x.rolling(window, min_periods=1).std())

    # 카테고리 인코딩
    print_progress("🔤 카테고리 변수 인코딩 중...", start_time)
    categorical_cols = ['city', 'sku', 'country', 'family', 'season']
    label_encoders = {}
    for col in categorical_cols:
        if col in demand.columns:
            le = LabelEncoder()
            demand[f'{col}_encoded'] = le.fit_transform(demand[col].astype(str))
            label_encoders[col] = le

    # 할인율 정규화 및 NaN 디버깅
    demand['discount_pct'] = demand['discount_pct'] / 100
    print("=== Before fillna ===")
    for col in ["demand","unit_price","discount_pct","spend_usd","brent_usd","confidence_index"]:
        if col in demand.columns:
            print(col, "nan:", demand[col].isna().sum(), "min:", demand[col].min() if demand[col].notna().any() else None)
    print(demand[demand["unit_price"].isna()].head(10)[["date","city","sku","unit_price"]])

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

def prepare_lightgbm_features(demand_data):
    """LightGBM 모델용 피처 준비 (동일 로직 재현)"""
    print_progress("🔧 LightGBM 모델용 피처 준비 중...")
    start_time = time.time()
    feature_cols = [
        'month', 'dayofyear', 'weekday',
        'storage_gb', 'days_since_launch',
        'city_encoded', 'sku_encoded', 'country_encoded', 'family_encoded', 'season_encoded',
        'unit_price', 'discount_pct',
        'avg_temp', 'humidity',
        'brent_usd', 'pct_change', 'volatility_7d',
        'confidence_index', 'spend_usd',
        'EUR=X', 'KRW=X', 'JPY=X', 'GBP=X', 'CAD=X', 'AUD=X', 'BRL=X', 'ZAR=X',
        'is_event'
    ]
    ts_features = [col for col in demand_data.columns if any(x in col for x in ['lag_', 'rolling_mean_', 'rolling_std_'])]
    feature_cols.extend(ts_features)
    feature_cols = [col for col in feature_cols if col in demand_data.columns]
    remove_features = ['dayofyear']
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
        X_temp = X.drop(columns=[feature])
        y_temp = X[feature]
        model = LinearRegression()
        model.fit(X_temp, y_temp)
        r_squared = model.score(X_temp, y_temp)
        if r_squared < 0.999:
            vif = 1 / (1 - r_squared)
        else:
            vif = float('inf')
        vif_data.append({'feature': feature, 'vif': vif, 'r_squared': r_squared})
    vif_df = pd.DataFrame(vif_data)
    vif_df = vif_df.sort_values('vif', ascending=False)
    return vif_df

def analyze_multicollinearity(X, feature_names, start_time):
    """다중공선성 분석 (VIF + 상관관계)"""
    print_progress("🔍 다중공선성 분석 시작...", start_time)
    vif_df = calculate_vif(X, feature_names)
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
    vif_csv_path = DATA_DIR / 'vif_analysis.csv'
    vif_df.to_csv(vif_csv_path, index=False)
    print_progress(f"📁 VIF 분석 결과 저장: {vif_csv_path}", start_time)
    print_progress("📊 상관관계 분석 중...", start_time)
    corr_matrix = X.corr()
    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            corr_val = corr_matrix.iloc[i, j]
            if abs(corr_val) > 0.8:
                high_corr_pairs.append({'feature1': corr_matrix.columns[i], 'feature2': corr_matrix.columns[j], 'correlation': corr_val})
    if high_corr_pairs:
        print("🔗 높은 상관관계 (|r| > 0.8):")
        high_corr_pairs.sort(key=lambda x: abs(x['correlation']), reverse=True)
        for pair in high_corr_pairs[:20]:
            print(f"  - {pair['feature1']} ↔ {pair['feature2']}: r={pair['correlation']:.4f}")
        print()
    plt.figure(figsize=(20, 16))
    mask = np.abs(corr_matrix) < 0.5
    sns.heatmap(corr_matrix, mask=mask, annot=False, cmap='RdBu_r', center=0, square=True, linewidths=0.5, cbar_kws={"shrink": .8})
    plt.title('Feature Correlation Matrix (|r| >= 0.5)', fontsize=16)
    plt.tight_layout()
    corr_png_path = DATA_DIR / 'feature_correlation_heatmap.png'
    plt.savefig(corr_png_path, dpi=300, bbox_inches='tight')
    plt.close()
    print_progress(f"📁 상관관계 히트맵 저장: {corr_png_path}", start_time)
    print_progress("📊 피처 그룹별 다중공선성 분석...", start_time)
    lag_features = [f for f in feature_names if 'lag_' in f]
    rolling_mean_features = [f for f in feature_names if 'rolling_mean_' in f]
    rolling_std_features = [f for f in feature_names if 'rolling_std_' in f]
    print(f"  - Lag 피처 수: {len(lag_features)}")
    print(f"  - Rolling Mean 피처 수: {len(rolling_mean_features)}")
    print(f"  - Rolling Std 피처 수: {len(rolling_std_features)}")
    for group_name, group_features in [('Lag', lag_features), ('Rolling Mean', rolling_mean_features), ('Rolling Std', rolling_std_features)]:
        if group_features:
            group_vif = vif_df[vif_df['feature'].isin(group_features)]
            high_vif_in_group = group_vif[group_vif['vif'] > 5]
            if not high_vif_in_group.empty:
                print(f"  - {group_name} 그룹 내 높은 VIF 피처:")
                for _, row in high_vif_in_group.iterrows():
                    print(f"    * {row['feature']}: VIF={row['vif']:.2f}")
    return vif_df, high_corr_pairs

def _compute_event_weights(train_data: pd.DataFrame, event_weight: float = 100) -> np.ndarray:
    """이벤트 구간 가중치 벡터 생성"""
    return np.where(train_data['is_event'] == 1, event_weight, 1.0)


def _plot_and_save_feature_importance(model: xgb.XGBRegressor, feature_cols, start_time):
    """XGBoost 피처 중요도 계산 및 저장/시각화"""
    try:
        print_progress("📊 피처 중요도 계산 중...", start_time)

        # 1) gain 기반 중요도 산출
        booster = model.get_booster()
        # 최신 버전은 booster.feature_names가 pandas 컬럼명을 보존함
        feature_names = booster.feature_names if getattr(booster, 'feature_names', None) else list(feature_cols)
        gain_map = booster.get_score(importance_type='gain')  # dict: {feature_name: gain}

        # dict를 DataFrame으로 정리 (존재하지 않는 피처는 0 처리)
        gain_values = []
        for f in feature_names:
            gain_values.append(gain_map.get(f, 0.0))

        fi_df = pd.DataFrame({
            'feature': feature_names,
            'gain': gain_values,
        })
        fi_df['gain_pct'] = fi_df['gain'] / (fi_df['gain'].sum() + 1e-9)
        fi_df = fi_df.sort_values('gain', ascending=False).reset_index(drop=True)

        # 저장
        fi_csv_path = DATA_DIR / 'xgboost_feature_importance.csv'
        fi_df.to_csv(fi_csv_path, index=False)
        print_progress(f"📁 피처 중요도 CSV 저장: {fi_csv_path}", start_time)

        # 상위 30개 시각화
        top_n = min(30, len(fi_df))
        top_df = fi_df.head(top_n).iloc[::-1]
        plt.figure(figsize=(10, max(6, top_n * 0.3)))
        plt.barh(top_df['feature'], top_df['gain'])
        plt.title('XGBoost Feature Importance (gain 기준)')
        plt.xlabel('Gain')
        plt.tight_layout()
        fi_png_path = DATA_DIR / 'xgboost_feature_importance_top30.png'
        plt.savefig(fi_png_path, dpi=300, bbox_inches='tight')
        plt.close()
        print_progress(f"📁 피처 중요도 그래프 저장: {fi_png_path}", start_time)

        # 콘솔 상위 20개 출력
        print("\n상위 중요 피처 (gain 기준) Top 20:")
        for i, row in fi_df.head(20).iterrows():
            print(f"  {i+1:2d}. {row['feature']}: gain={row['gain']:.1f}, gain_pct={row['gain_pct']*100:.2f}%")
        print()
    except Exception as e:
        print(f"⚠️ 피처 중요도 계산/저장 중 오류: {e}")


def train_xgboost_model(train_data: pd.DataFrame, val_data: pd.DataFrame, feature_cols, start_time):
    """XGBoost 모델 학습 (GPU 우선, 실패 시 CPU 폴백)"""
    print_progress("🚀 XGBoost 모델 학습 시작...", start_time)

    # 데이터 준비
    X_train = train_data[feature_cols]
    y_train = train_data['demand']
    X_val = val_data[feature_cols]
    y_val = val_data['demand']

    print_progress(f"📊 학습 데이터: {X_train.shape}, 검증 데이터: {X_val.shape}", start_time)

    # 이벤트 가중치
    print_progress("🔍 디버깅: 이벤트 데이터 분석 중...", start_time)
    train_event_count = train_data['is_event'].sum()
    train_total_count = len(train_data)
    val_event_count = val_data['is_event'].sum()
    val_total_count = len(val_data)
    print(f"  - 훈련 데이터: 총 {train_total_count:,}개 중 이벤트 {train_event_count:,}개 ({train_event_count/train_total_count*100:.2f}%)")
    print(f"  - 검증 데이터: 총 {val_total_count:,}개 중 이벤트 {val_event_count:,}개 ({val_event_count/val_total_count*100:.2f}%)")

    event_weight = 100
    final_weights = _compute_event_weights(train_data, event_weight)
    print(f"  - 이벤트 가중치: {event_weight}")
    print(f"  - 이벤트 기간 평균 가중치: {final_weights[train_data['is_event'] == 1].mean():.1f}")
    print(f"  - 비이벤트 기간 평균 가중치: {final_weights[train_data['is_event'] == 0].mean():.1f}")
    print(f"  - 가중치 비율: {final_weights[train_data['is_event'] == 1].mean() / final_weights[train_data['is_event'] == 0].mean():.1f}배")

    # 공통 하이퍼파라미터 (LightGBM 설정을 XGBoost에 유사 매핑)
    # 버전에 따라 GPU 파라미터를 안정적으로 설정
    _ver = getattr(xgb, '__version__', '1.6.0')
    try:
        _ver_parts = [int(p) for p in _ver.split('.')[:2]]
    except Exception:
        _ver_parts = [1, 6]
    _major, _minor = (_ver_parts + [0, 0])[:2]

    gpu_params = {}
    if _major >= 2:
        # XGBoost 2.x: device 파라미터 사용 + GPU predictor 강제
        gpu_params = dict(device='cuda', tree_method='hist', predictor='gpu_predictor')
    else:
        # XGBoost 1.x: gpu_hist / gpu_predictor 사용
        gpu_params = dict(tree_method='gpu_hist', predictor='gpu_predictor')

    common_params = dict(
        n_estimators=1200,
        learning_rate=0.005,
        max_depth=7,
        subsample=0.7,
        colsample_bytree=0.8,
        min_child_weight=1.0,
        gamma=0.0,
        reg_alpha=0.0,
        reg_lambda=1.0,
        random_state=42,
        objective='reg:squarederror',
        eval_metric='rmse',
        n_jobs=-1,
        verbosity=1,
        **gpu_params,
    )

    # 학습 (GPU → 실패 시 CPU로 재시도)
    def _fit_with_params(params):
        model = xgb.XGBRegressor(**params)
        # 호환성: 일부 버전은 early_stopping_rounds/verbose/callbacks 미지원
        try:
            try:
                # 우선 callbacks 방식 시도
                es_cb = getattr(xgb.callback, 'EarlyStopping', None)
                callbacks = [es_cb(rounds=50)] if es_cb is not None else None
                model.fit(
                    X_train,
                    y_train,
                    sample_weight=final_weights,
                    eval_set=[(X_train, y_train), (X_val, y_val)],
                    callbacks=callbacks,
                )
            except TypeError:
                # callbacks 인자 미지원 → 단순 학습 (ES 비활성화)
                print_progress("⚠️ 현재 XGBoost 버전에서 Early Stopping 미지원 → ES 비활성화하고 학습 진행", start_time)
                model.fit(
                    X_train,
                    y_train,
                    sample_weight=final_weights,
                    eval_set=[(X_train, y_train), (X_val, y_val)],
                )
        except TypeError as e:
            # 최후 수단: eval_set 도 제거
            print_progress(f"⚠️ fit 인자 호환성 문제로 eval_set 제거 후 재시도: {e}", start_time)
            model.fit(
                X_train,
                y_train,
                sample_weight=final_weights,
            )
        return model

    print_progress("📚 모델 학습 중... (GPU 시도)", start_time)
    # 빌드 설정(가능 시) 출력: CUDA 지원 여부 확인에 도움
    try:
        if hasattr(xgb, 'print_config'):
            print_progress("🔧 XGBoost Build Config:", start_time)
            xgb.print_config()
    except Exception as e:
        print(f"[WARN] print_config 실패: {e}")
    try:
        model = _fit_with_params(common_params)
    except Exception as gpu_err:
        print(f"⚠️ GPU 학습 실패, CPU로 폴백합니다: {gpu_err}")
        cpu_params = {**common_params, 'tree_method': 'hist', 'predictor': 'auto'}
        model = _fit_with_params(cpu_params)

    print_progress(f"✅ XGBoost 모델 학습 완료: {time.time() - start_time:.1f}초", start_time)

    # 피처 중요도 산출 및 저장
    _plot_and_save_feature_importance(model, feature_cols, start_time)

    # 실제 사용된 장치/알고리즘 확인 (Booster 설정 + 모델 파라미터)
    try:
        import json as _json
        cfg_json = model.get_booster().save_config()
        cfg = _json.loads(cfg_json)
        generic = cfg.get('learner', {}).get('generic_param', {})
        used_tree_method = str(generic.get('tree_method', 'unknown'))
        used_predictor = str(generic.get('predictor', 'unknown'))
        device = str(generic.get('device', 'n/a'))

        # 모델 파라미터도 함께 확인
        params_used = model.get_xgb_params()
        p_tree_method = str(params_used.get('tree_method', ''))
        p_predictor = str(params_used.get('predictor', ''))
        p_device = str(params_used.get('device', ''))

        print_progress(
            f"🧭 Booster Config → tree_method={used_tree_method}, predictor={used_predictor}, device={device}",
            start_time
        )
        print_progress(
            f"🧭 Model Params  → tree_method={p_tree_method}, predictor={p_predictor}, device={p_device}",
            start_time
        )

        used_gpu = (
            'cuda' in device.lower() or 'cuda' in p_device.lower() or
            'gpu' in used_predictor.lower() or 'gpu' in p_predictor.lower() or
            used_tree_method == 'gpu_hist' or p_tree_method == 'gpu_hist'
        )
        if used_gpu:
            print_progress("✅ GPU 사용으로 판단됨", start_time)
        else:
            print_progress("⚠️ GPU 사용 징후가 없어 CPU로 판단됨", start_time)
    except Exception as e:
        print(f"[WARN] Booster 설정 확인 실패: {e}")

    return model


def create_xgboost_validation_visualization(val_data: pd.DataFrame, val_pred: np.ndarray, start_time):
    """XGBoost 모델 2022년 검증 결과 시각화"""
    print_progress("📊 XGBoost 모델 2022년 검증 시각화 생성 중...", start_time)

    val_data_with_pred = val_data.copy()
    val_data_with_pred['predicted_demand'] = val_pred

    # 샘플링 (한국 4개 도시 고정, 없으면 임의 상위 5 조합)
    korean_cities = ['Seoul', 'Busan', 'Incheon', 'Gwangju']
    sample_combinations = []
    for city in korean_cities:
        city_data = val_data_with_pred[val_data_with_pred['city'] == city]
        if not city_data.empty:
            first_sku = city_data['sku'].iloc[0]
            sample_combinations.append({'city': city, 'sku': first_sku})
    if len(sample_combinations) == 0:
        sample_combinations = val_data_with_pred[['city', 'sku']].drop_duplicates().head(5).to_dict('records')

    fig, axes = plt.subplots(len(sample_combinations), 1, figsize=(15, 4 * len(sample_combinations)))
    if len(sample_combinations) == 1:
        axes = [axes]

    for i, combo in enumerate(sample_combinations):
        city, sku = combo['city'], combo['sku']
        mask = (val_data_with_pred['city'] == city) & (val_data_with_pred['sku'] == sku)
        combo_data = val_data_with_pred[mask].sort_values('date')
        if len(combo_data) > 0:
            actual_vals = combo_data['demand'].values
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
    output_path = DATA_DIR / "xgboost_validation_2022.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print_progress(f"📁 Saved XGBoost 2022 validation plot: {output_path}", start_time)


def create_xgboost_full_timeline_visualization(demand_data: pd.DataFrame, result_df: pd.DataFrame, start_time):
    """XGBoost 모델 2018-2024 전체 시계열 예측 시각화"""
    print_progress("📊 Creating XGBoost 2018-2024 full timeline plot...", start_time)

    print_progress("🔍 Debug: Checking stats for full timeline visualization...", start_time)
    print(f"  - 2023-2024 예측값 (result_df): Mean={result_df['mean'].mean():.2f}, Max={result_df['mean'].max():.2f}, Non-zero ratio={(result_df['mean'] > 0).mean()*100:.2f}%")

    actual_data = demand_data[['date', 'city', 'sku', 'demand']].copy()
    actual_data['type'] = 'actual'
    pred_vis_data = result_df[['date', 'city', 'sku', 'mean']].copy().rename(columns={'mean': 'demand'})
    pred_vis_data['type'] = 'predicted'
    combined_data = pd.concat([
        actual_data[['date', 'city', 'sku', 'demand', 'type']],
        pred_vis_data
    ], ignore_index=True)
    combined_data['date'] = pd.to_datetime(combined_data['date'])

    available_cities = combined_data['city'].unique()
    available_skus = combined_data['sku'].unique()
    sample_cities = available_cities[:5] if len(available_cities) >= 5 else available_cities
    sample_skus = available_skus[:3] if len(available_skus) >= 3 else available_skus

    print(f"📊 Sample cities for plotting: {sample_cities}")
    print(f"📊 Sample SKUs for plotting: {sample_skus}")

    fig, axes = plt.subplots(len(sample_cities), len(sample_skus), figsize=(24, 18))
    fig.suptitle('XGBoost: 2018-2024 Full Timeline - Actual vs Predicted', fontsize=16, fontweight='bold')

    for i, city in enumerate(sample_cities):
        for j, sku in enumerate(sample_skus):
            city_sku_data = combined_data[(combined_data['city'] == city) & (combined_data['sku'] == sku)]
            if len(city_sku_data) > 0:
                city_sku_data = city_sku_data.sort_values('date')
                actual_mask = city_sku_data['type'] == 'actual'
                pred_mask = city_sku_data['type'] == 'predicted'
                actual_plot = city_sku_data[actual_mask]
                pred_plot = city_sku_data[pred_mask]

                if len(actual_plot) > 0:
                    axes[i, j].plot(actual_plot['date'], actual_plot['demand'], label='Actual (2018-2022)', color='blue', linewidth=2)
                if len(pred_plot) > 0:
                    axes[i, j].plot(pred_plot['date'], pred_plot['demand'], label='Predicted (2023-2024)', color='red', linewidth=2, linestyle='--')

                axes[i, j].axvline(x=pd.Timestamp('2023-01-01'), color='green', linestyle=':', alpha=0.7, label='Forecast start')
                axes[i, j].set_title(f'{city} - {sku}', fontsize=12, fontweight='bold')
                axes[i, j].set_xlabel('Date')
                axes[i, j].set_ylabel('Demand')
                axes[i, j].legend()
                axes[i, j].grid(True, alpha=0.3)
                axes[i, j].tick_params(axis='x', rotation=45)
                if len(actual_plot) > 0 and len(pred_plot) > 0:
                    y_min = min(actual_plot['demand'].min(), pred_plot['demand'].min())
                    y_max = max(actual_plot['demand'].max(), pred_plot['demand'].max())
                    axes[i, j].set_ylim([y_min * 0.8, y_max * 1.2])
            else:
                axes[i, j].text(0.5, 0.5, '데이터 없음', ha='center', va='center', transform=axes[i, j].transAxes)
                axes[i, j].set_title(f'{city} - {sku}', fontsize=12, fontweight='bold')

    plt.tight_layout()
    output_path = DATA_DIR / "xgboost_full_timeline_2018_2024.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print_progress(f"📁 Saved XGBoost full timeline plot: {output_path}", start_time)


## 예측 헬퍼는 원래 방식으로 복구했으므로 제거

def predict_future_xgboost(model, demand_data, feature_cols, label_encoders, start_time, events_df=None):
    """XGBoost 모델로 미래 예측 (final_forecast_model의 로직을 동일하게 유지)"""
    print_progress("🔮 XGBoost 모델로 미래 예측 중...", start_time)

    # 2023-2024년 날짜 생성 및 공변량/이벤트/오토레그 로직은 그대로 재사용하기 위해
    # final_forecast_model의 구현을 가져오지 않고 동일 구현을 이곳에 복제
    future_dates = pd.date_range(start='2023-01-01', end='2024-12-31', freq='D')

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

    marketing_agg = marketing.groupby(['date', 'country'])['spend_usd'].sum().reset_index()
    country_cov = consumer_conf[['date', 'country', 'confidence_index']]
    country_cov = country_cov.merge(marketing_agg, on=['date', 'country'], how='left')
    country_cov = country_cov.merge(weather[['date', 'country', 'avg_temp', 'humidity']], on=['date', 'country'], how='left')
    country_cov = country_cov.merge(calendar[['date', 'country', 'season']], on=['date', 'country'], how='left')

    cov_date = oil[['date', 'brent_usd', 'pct_change', 'volatility_7d']]
    cov_date = cov_date.merge(currency[['date'] + fx_cols], on='date', how='left')

    cov_future = country_cov.merge(cov_date, on='date', how='left')

    # 결측/미존재 값 보정: 미래(2023-2024) 공변량이 비면 0으로 떨어질 수 있음 → 그룹별/시간별 보간
    try:
        cov_future = cov_future.sort_values(['country', 'date'])
        cov_future[['confidence_index','spend_usd','avg_temp','humidity']] = (
            cov_future.groupby('country')[['confidence_index','spend_usd','avg_temp','humidity']].ffill()
        )
        # 남은 결측은 0이 아닌 중앙값으로 채움
        for c in ['confidence_index','spend_usd','avg_temp','humidity']:
            if c in cov_future.columns:
                med = cov_future[c].median() if cov_future[c].notna().any() else 0.0
                cov_future[c] = cov_future[c].fillna(med)

        # season 문자열 결측은 직전값으로 유지
        if 'season' in cov_future.columns:
            cov_future['season'] = cov_future.groupby('country')['season'].ffill().fillna('')

        # 날짜 공변량도 시간 축으로 보간
        cov_date = cov_date.sort_values('date')
        fill_cols = ['brent_usd','pct_change','volatility_7d'] + fx_cols
        for c in fill_cols:
            if c in cov_date.columns:
                cov_date[c] = cov_date[c].ffill()
    except Exception as e:
        print(f"[WARN] cov fill forward failed: {e}")

    # 디버깅: cov 테이블 구조 미리보기
    try:
        print_progress("🔎 cov_future preview:", start_time)
        print(f"  - shape: {cov_future.shape}")
        print(f"  - columns: {list(cov_future.columns)}")
        # 2023~2024 결측률 요약
        mask_2324 = (cov_future['date'] >= pd.Timestamp('2023-01-01')) & (cov_future['date'] <= pd.Timestamp('2024-12-31'))
        sub = cov_future.loc[mask_2324]
        if len(sub) > 0:
            check_cols = ['confidence_index','spend_usd','avg_temp','humidity','brent_usd','pct_change','volatility_7d']
            nz = {c: float((sub[c]==0).mean()) if c in sub.columns else None for c in check_cols}
            print(f"  - 2023-2024 zero ratio: {nz}")
        print(cov_future.head(5))
    except Exception as e:
        print(f"[WARN] cov_future preview failed: {e}")

    event_periods = get_hardcoded_event_periods()
    print_progress("📢 2023-2024 확정 이벤트 구간:")
    for (country, year), (start_date, end_date) in event_periods.items():
        if year in [2023, 2024]:
            print(f"  - {country} | {start_date} ~ {end_date}")

    city_sku_combinations = demand_data[['city', 'sku']].drop_duplicates()
    result = []
    print_progress(f"📊 예측 대상: {len(city_sku_combinations)}개 조합 × {len(future_dates)}일 = {len(city_sku_combinations) * len(future_dates):,}개", start_time)
    # 예측 전 GPU 경로 재확인 (predictor/device)
    try:
        import json as _json
        cfg_json = model.get_booster().save_config()
        cfg = _json.loads(cfg_json)
        generic = cfg.get('learner', {}).get('generic_param', {})
        used_predictor = str(generic.get('predictor', 'unknown'))
        device = str(generic.get('device', 'n/a'))
        print_progress(f"🔎 Predict Config → predictor={used_predictor}, device={device}", start_time)
    except Exception as e:
        print(f"[WARN] 예측 전 GPU 설정 확인 실패: {e}")

    for idx, (_, combo) in enumerate(tqdm(city_sku_combinations.iterrows(), total=len(city_sku_combinations), desc="예측 진행")):
        city, sku = combo['city'], combo['sku']
        recent_data = demand_data[(demand_data['city'] == city) & (demand_data['sku'] == sku)].tail(60)
        if len(recent_data) == 0:
            continue

        demand_buffer = recent_data['demand'].dropna().tolist()
        if len(demand_buffer) == 0:
            demand_buffer = [0.0]
        if idx < 3:
            print(f"[DBG] {city}-{sku} initial demand_buffer (last 5): {demand_buffer[-5:]}  size={len(demand_buffer)}")

        for date in future_dates:
            future_row = {
                'date': date,
                'city': city,
                'sku': sku,
                'month': date.month,
                'dayofyear': date.dayofyear,
                'weekday': date.weekday(),
            }

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

            lookup_country = future_row['country'] if 'country' in future_row else None
            cov_row = cov_future[(cov_future['date'] == date) & (cov_future['country'] == lookup_country)]
            if len(cov_row) > 0:
                cov_row = cov_row.iloc[0]
                future_row['confidence_index'] = cov_row.get('confidence_index', 0)
                future_row['spend_usd'] = cov_row.get('spend_usd', 0)
                future_row['avg_temp'] = cov_row.get('avg_temp', 0)
                future_row['humidity'] = cov_row.get('humidity', 0)
                future_row['season'] = cov_row.get('season', future_row.get('season', ''))
            else:
                for col in ['confidence_index', 'spend_usd', 'avg_temp', 'humidity']:
                    future_row[col] = 0

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

            if sku in sku_meta['sku'].values:
                sku_launch_date = sku_meta[sku_meta['sku'] == sku]['launch_date'].iloc[0]
                if pd.notna(sku_launch_date):
                    future_row['days_since_launch'] = max(0, (date - sku_launch_date).days)
                else:
                    future_row['days_since_launch'] = 0
            else:
                future_row['days_since_launch'] = 0

            for col in ['city', 'sku', 'country', 'family', 'season']:
                # 인코더는 상위 레벨에서 주입되므로 여기서는 열만 준비
                pass

            # 이벤트 플래그
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

            if 'unit_price' in recent_data.columns and len(recent_data) > 0:
                future_row['unit_price'] = float(recent_data['unit_price'].ffill().iloc[-1])
            if 'discount_pct' in recent_data.columns and len(recent_data) > 0:
                future_row['discount_pct'] = float(recent_data['discount_pct'].ffill().iloc[-1])
                # 할인율 스케일 보정(학습과 동일 0~1)
                if future_row['discount_pct'] > 1.0:
                    future_row['discount_pct'] = future_row['discount_pct'] / 100.0

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

            row_df = pd.DataFrame([future_row])
            # 누락된 피처 보정
            for col in feature_cols:
                if col not in row_df.columns:
                    row_df[col] = 0

            # 카테고리 인코딩 (학습 시 LabelEncoder 사용과 동일하게 처리)
            for col in ['city', 'sku', 'country', 'family', 'season']:
                encoded_col = f"{col}_encoded"
                if encoded_col in feature_cols:
                    value_to_encode = str(future_row.get(col, ''))
                    if value_to_encode is None:
                        value_to_encode = ''
                    if col in label_encoders:
                        le = label_encoders[col]
                        # 미지의 클래스 처리: 첫 클래스 대체
                        if len(le.classes_) > 0 and value_to_encode not in le.classes_:
                            value_to_encode = str(le.classes_[0])
                        row_df[encoded_col] = le.transform([value_to_encode])[0]
                    else:
                        row_df[encoded_col] = 0

            X_row = row_df[feature_cols]
            pred_demand = float(model.predict(X_row)[0])
            pred_demand = int(max(0, round(pred_demand)))

            future_row['mean'] = pred_demand
            result.append(future_row)

            demand_buffer.append(pred_demand)
            if len(demand_buffer) > 60:
                demand_buffer = demand_buffer[-60:]

    result_df = pd.DataFrame(result)[['sku', 'city', 'date', 'mean']]

    # 연속성 보정: 2022-12 최근 수요 대비 2023-01 초 예측이 과도하게 낮아지는 현상 완화
    try:
        last_ref = demand_data[(demand_data['date'] >= '2022-12-01') & (demand_data['date'] <= '2022-12-31')]
        ref_mean = last_ref.groupby(['city','sku'])['demand'].mean().rename('ref_mean')

        first_pred = result_df[result_df['date'] <= pd.Timestamp('2023-01-07')]
        pred_mean = first_pred.groupby(['city','sku'])['mean'].mean().rename('pred_mean')

        scale_df = ref_mean.reset_index().merge(pred_mean.reset_index(), on=['city','sku'], how='left')
        scale_df['pred_mean'] = scale_df['pred_mean'].fillna(scale_df['pred_mean'].median() if scale_df['pred_mean'].notna().any() else 1.0)
        scale_df['scale'] = scale_df['ref_mean'] / scale_df['pred_mean'].clip(lower=1e-6)
        # 과도한 스케일은 클리핑
        scale_df['scale'] = scale_df['scale'].clip(lower=0.8, upper=1.5)

        result_df = result_df.merge(scale_df[['city','sku','scale']], on=['city','sku'], how='left')
        result_df['scale'] = result_df['scale'].fillna(1.0)
        result_df['mean'] = (result_df['mean'] * result_df['scale']).round().astype(int)
        result_df = result_df.drop(columns=['scale'])
        print_progress("🔧 적용: 2023-01 연속성 보정(스케일링 0.5~2.0)", start_time)
    except Exception as e:
        print(f"[WARN] 연속성 보정 단계 실패: {e}")

    print_progress(f"📊 예측값 통계 - 평균: {result_df['mean'].mean():.2f}, 중앙값: {result_df['mean'].median():.2f}, 최대: {result_df['mean'].max():.2f}, 최소: {result_df['mean'].min():.2f}", start_time)
    print_progress(f"📊 0이 아닌 예측값 비율: {(result_df['mean'] > 0).mean():.3f}", start_time)
    return result_df


def predict_future_xgboost_batched(model, demand_data, feature_cols, label_encoders, start_time, events_df=None):
    """XGBoost 모델로 미래 예측 (날짜 단위 배치 예측으로 속도 최적화)"""
    print_progress("🔮 XGBoost 모델로 미래 예측 중... (배치 예측)", start_time)

    # 2023-2024 날짜 생성
    future_dates = pd.date_range(start='2023-01-01', end='2024-12-31', freq='D')

    # 외생/보조 피처 로드 (train과 동일 전처리: pct_change/volatility_7d, 할인율 스케일 등)
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

    marketing_agg = marketing.groupby(['date', 'country'])['spend_usd'].sum().reset_index()
    country_cov = consumer_conf[['date', 'country', 'confidence_index']]
    country_cov = country_cov.merge(marketing_agg, on=['date', 'country'], how='left')
    country_cov = country_cov.merge(weather[['date', 'country', 'avg_temp', 'humidity']], on=['date', 'country'], how='left')
    country_cov = country_cov.merge(calendar[['date', 'country', 'season']], on=['date', 'country'], how='left')

    cov_date = oil[['date', 'brent_usd', 'pct_change', 'volatility_7d']]
    cov_date = cov_date.merge(currency[['date'] + fx_cols], on='date', how='left')

    cov_future = country_cov.merge(cov_date, on='date', how='left')

    try:
        print_progress("🔎 cov_future preview:", start_time)
        print(f"  - shape: {cov_future.shape}")
        print(f"  - columns: {list(cov_future.columns)}")
        print(cov_future.head(5))
    except Exception as e:
        print(f"[WARN] cov_future preview failed: {e}")

    event_periods = get_hardcoded_event_periods()
    print_progress("📢 2023-2024 확정 이벤트 구간:", start_time)
    for (country, year), (start_date, end_date) in event_periods.items():
        if year in [2023, 2024]:
            print(f"  - {country} | {start_date} ~ {end_date}")

    # 모든 조합
    city_sku_combinations = demand_data[['city', 'sku']].drop_duplicates().reset_index(drop=True)
    result = []
    print_progress(f"📊 예측 대상: {len(city_sku_combinations)}개 조합 × {len(future_dates)}일 = {len(city_sku_combinations) * len(future_dates):,}개", start_time)

    # 인덱싱 최적화
    cov_future_idx = cov_future.set_index(['date', 'country']).sort_index()
    cov_date_idx = cov_date.set_index('date').sort_index()
    sku_launch_map = {row['sku']: row['launch_date'] for _, row in sku_meta.iterrows()}

    # 조합별 버퍼/최근값/메타 준비
    buffers = {}
    recent_map = {}
    meta_map = {}
    for idx, (_, combo) in enumerate(city_sku_combinations.iterrows()):
        city, sku = combo['city'], combo['sku']
        recent_data = demand_data[(demand_data['city'] == city) & (demand_data['sku'] == sku)].tail(60)
        if len(recent_data) == 0:
            continue
        buf = recent_data['demand'].dropna().tolist() or [0.0]
        buffers[(city, sku)] = buf
        recent_map[(city, sku)] = recent_data
        meta_map[(city, sku)] = {
            'country': recent_data['country'].iloc[0] if 'country' in recent_data.columns else None,
            'family': recent_data['family'].iloc[0] if 'family' in recent_data.columns else None,
            'season': recent_data['season'].iloc[0] if 'season' in recent_data.columns else None,
        }
        if idx < 3:
            print(f"[DBG] {city}-{sku} initial demand_buffer (last 5): {buf[-5:]}  size={len(buf)}")

    # 예측 전 GPU 경로 재확인
    try:
        import json as _json
        cfg_json = model.get_booster().save_config()
        cfg = _json.loads(cfg_json)
        generic = cfg.get('learner', {}).get('generic_param', {})
        used_predictor = str(generic.get('predictor', 'unknown'))
        device = str(generic.get('device', 'n/a'))
        print_progress(f"🔎 Predict Config → predictor={used_predictor}, device={device}", start_time)
    except Exception as e:
        print(f"[WARN] 예측 전 GPU 설정 확인 실패: {e}")

    # 날짜 단위 배치 예측
    for date in tqdm(future_dates, desc="예측 진행(날짜 배치)"):
        batch_rows = []
        batch_keys = []

        for _, combo in city_sku_combinations.iterrows():
            city, sku = combo['city'], combo['sku']
            key = (city, sku)
            if key not in buffers:
                continue
            recent_data = recent_map[key]
            buf = buffers[key]
            meta = meta_map[key]
            country = meta['country']
            family = meta['family']
            season_fixed = meta['season']

            future_row = {
                'date': date,
                'city': city,
                'sku': sku,
                'month': date.month,
                'dayofyear': date.dayofyear,
                'weekday': date.weekday(),
                'country': country,
                'family': family,
                'season': season_fixed,
            }

            # 국가/날짜 공변량
            try:
                cov_row = cov_future_idx.loc[(date, country)]
                future_row['confidence_index'] = cov_row.get('confidence_index', 0)
                future_row['spend_usd'] = cov_row.get('spend_usd', 0)
                future_row['avg_temp'] = cov_row.get('avg_temp', 0)
                future_row['humidity'] = cov_row.get('humidity', 0)
                future_row['season'] = cov_row.get('season', future_row.get('season', ''))
            except Exception:
                for col in ['confidence_index', 'spend_usd', 'avg_temp', 'humidity']:
                    future_row[col] = 0

            try:
                cov_date_row = cov_date_idx.loc[date]
                future_row['brent_usd'] = cov_date_row.get('brent_usd', 0)
                future_row['pct_change'] = cov_date_row.get('pct_change', 0)
                future_row['volatility_7d'] = cov_date_row.get('volatility_7d', 0)
                for fx in fx_cols:
                    future_row[fx] = cov_date_row.get(fx, 0)
            except Exception:
                for col in ['brent_usd', 'pct_change', 'volatility_7d'] + fx_cols:
                    future_row[col] = 0

            # 출시일 경과일
            launch_date = sku_launch_map.get(sku, None)
            if launch_date is not None and pd.notna(launch_date):
                future_row['days_since_launch'] = max(0, (date - launch_date).days)
            else:
                future_row['days_since_launch'] = 0

            # 이벤트 플래그
            is_event = 0
            for (cty, year), (start_date, end_date) in event_periods.items():
                if cty == country and date >= pd.to_datetime(start_date) and date <= pd.to_datetime(end_date):
                    is_event = 1
                    break
            future_row['is_event'] = is_event

            # 수요 시계열 피처 (버퍼 기반) - 완전 0 버퍼 방지 가드
            if not buf:
                buf = [max(1.0, recent_data['demand'].mean() if len(recent_data) > 0 else 1.0)]
                buffers[key] = buf
            for lag in [1, 3, 7, 14]:
                future_row[f'demand_lag_{lag}'] = buf[-lag] if len(buf) >= lag else (buf[0] if len(buf) > 0 else 0)
            for window in [7, 14]:
                if len(buf) > 0:
                    series = buf[-window:] if len(buf) >= window else buf
                    future_row[f'demand_rolling_mean_{window}'] = float(np.mean(series))
                    future_row[f'demand_rolling_std_{window}'] = float(np.std(series, ddof=1)) if len(series) > 1 else 0.0
                else:
                    future_row[f'demand_rolling_mean_{window}'] = 0.0
                    future_row[f'demand_rolling_std_{window}'] = 0.0

            # 가격/할인 최근값 및 시계열
            if 'unit_price' in recent_data.columns and len(recent_data) > 0:
                future_row['unit_price'] = float(recent_data['unit_price'].ffill().iloc[-1])
            if 'discount_pct' in recent_data.columns and len(recent_data) > 0:
                future_row['discount_pct'] = float(recent_data['discount_pct'].ffill().iloc[-1])
                if future_row['discount_pct'] > 1.0:
                    future_row['discount_pct'] = future_row['discount_pct'] / 100.0
            for col in ['discount_pct', 'unit_price']:
                if col in recent_data.columns:
                    for lag in [1, 3, 7, 14]:
                        future_row[f'{col}_lag_{lag}'] = (
                            recent_data[col].iloc[-lag] if len(recent_data) >= lag else (recent_data[col].iloc[0] if len(recent_data) > 0 else 0)
                        )
                    for window in [7, 14]:
                        vals = recent_data[col].tail(window)
                        future_row[f'{col}_rolling_mean_{window}'] = vals.mean() if len(vals) > 0 else (recent_data[col].mean() if len(recent_data) > 0 else 0)

            # spend_usd lag/rolling
            for lag in [1, 3, 7, 14]:
                lag_date = date - pd.Timedelta(days=lag)
                try:
                    v = cov_future_idx.loc[(lag_date, country), 'spend_usd']
                    future_row[f'spend_usd_lag_{lag}'] = float(v) if pd.notna(v) else 0.0
                except Exception:
                    future_row[f'spend_usd_lag_{lag}'] = 0.0
            for window in [7, 14]:
                start_d = date - pd.Timedelta(days=window-1)
                try:
                    vals = cov_future_idx.loc[(slice(start_d, date), country), 'spend_usd']
                    future_row[f'spend_usd_rolling_mean_{window}'] = float(vals.mean()) if len(vals) > 0 else 0.0
                except Exception:
                    future_row[f'spend_usd_rolling_mean_{window}'] = 0.0

            # 카테고리 인코딩
            for col in ['city', 'sku', 'country', 'family', 'season']:
                encoded_col = f"{col}_encoded"
                if encoded_col in feature_cols:
                    val = str(future_row.get(col, ''))
                    if col in label_encoders and len(label_encoders[col].classes_) > 0:
                        if val not in label_encoders[col].classes_:
                            val = str(label_encoders[col].classes_[0])
                        future_row[encoded_col] = label_encoders[col].transform([val])[0]
                    else:
                        future_row[encoded_col] = 0

            batch_rows.append(future_row)
            batch_keys.append(key)

        if not batch_rows:
            continue

        batch_df = pd.DataFrame(batch_rows)
        # 디버그: 초기 1주 구간, 상위 3개 조합 샘플의 핵심 피처 상태 출력
        if date <= pd.Timestamp('2023-01-07'):
            sample_df = batch_df.head(3).copy()
            check_cols = [c for c in feature_cols if (
                c.startswith('demand_lag_') or c.startswith('demand_rolling_mean_') or
                c in ['unit_price','discount_pct','spend_usd','confidence_index','brent_usd']
            )]
            common = [c for c in check_cols if c in sample_df.columns]
            if len(sample_df) and len(common):
                nz_ratio = (sample_df[common].replace(0, np.nan).notna().mean(axis=1)).round(2).tolist()
                print_progress(f"[DBG] {str(date.date())} first3 nz_ratio={nz_ratio}", start_time)
                try:
                    for r in range(min(3, len(sample_df))):
                        cid, sk = sample_df.iloc[r]['city'], sample_df.iloc[r]['sku']
                        dl1 = sample_df.iloc[r].get('demand_lag_1', np.nan)
                        su = sample_df.iloc[r].get('spend_usd', np.nan)
                        up = sample_df.iloc[r].get('unit_price', np.nan)
                        dc = sample_df.iloc[r].get('discount_pct', np.nan)
                        print(f"  - {cid}-{sk} dlag1={dl1} spend={su} unit={up} disc={dc}")
                except Exception:
                    pass
        for col in feature_cols:
            if col not in batch_df.columns:
                batch_df[col] = 0
        X_batch = batch_df[feature_cols]
        preds = model.predict(X_batch)
        # 완전 0 드리프트 방지: 매우 작은 예측은 최근 평균의 작은 비율로 바닥 설정
        # 최근값 기준 바닥값 맵 구성
        floor_map = {}
        for key, recent_df in recent_map.items():
            m = float(recent_df['demand'].tail(14).mean()) if len(recent_df) > 0 else 0.0
            floor_map[key] = max(0.0, m * 0.10)  # 최근 10%를 바닥으로 (강화)

        for i, pred in enumerate(preds):
            city, sku = batch_keys[i]
            base_floor = floor_map.get((city, sku), 0.0)
            pred_adj = max(float(pred), base_floor)
            pred_demand = int(max(0, round(pred_adj)))
            result.append({
                'sku': sku,
                'city': city,
                'date': batch_rows[i]['date'],
                'mean': pred_demand,
            })
            buffers[(city, sku)].append(pred_demand)
            if len(buffers[(city, sku)]) > 60:
                buffers[(city, sku)] = buffers[(city, sku)][-60:]

    result_df = pd.DataFrame(result)[['sku', 'city', 'date', 'mean']]
    print_progress(f"📊 예측값 통계 - 평균: {result_df['mean'].mean():.2f}, 중앙값: {result_df['mean'].median():.2f}, 최대: {result_df['mean'].max():.2f}, 최소: {result_df['mean'].min():.2f}", start_time)
    print_progress(f"📊 0이 아닌 예측값 비율: {(result_df['mean'] > 0).mean():.3f}", start_time)
    return result_df


def generate_xgboost_forecast():
    """XGBoost 모델 기반 예측 생성 (LightGBM 로직과 동일 흐름)"""
    print_progress("=== XGBoost 모델 기반 고급 예측 생성 ===")
    total_start_time = time.time()

    # 1. 데이터 로드
    demand_data, events_df, label_encoders = load_enhanced_training_data()

    if events_df is not None and len(events_df) > 0:
        print("\n📢 확정된 이벤트 구간 목록 (2018~2024):")
        events_df_sorted = events_df.sort_values(['year','country','start_date']) if 'year' in events_df.columns else events_df
        for _, ev in events_df_sorted.iterrows():
            yr = ev['start_date'].year
            if 2018 <= yr <= 2024:
                print(f"  - {ev['country']} | {ev['start_date'].date()} ~ {ev['end_date'].date()} (year={yr})")

    # 2. 피처 준비 (동일 피처셋 재사용)
    demand_data, feature_cols = prepare_lightgbm_features(demand_data)

    # 3. 데이터 분할
    print_progress("📊 데이터 분할 중...", total_start_time)
    train_data = demand_data[demand_data['date'] < '2022-01-01'].copy()
    val_data = demand_data[(demand_data['date'] >= '2022-01-01') & (demand_data['date'] < '2023-01-01')].copy()
    print_progress(f"📊 데이터 분할 완료 - Train: {len(train_data):,}개, Val: {len(val_data):,}개", total_start_time)

    # 4. 다중공선성 분석 (동일 유지)
    print_progress("🔍 다중공선성 분석 중...", total_start_time)
    X_train_sample = train_data[feature_cols].sample(n=min(10000, len(train_data)), random_state=42)
    _ = analyze_multicollinearity(X_train_sample, feature_cols, total_start_time)

    # 5. 모델 학습
    model = train_xgboost_model(train_data, val_data, feature_cols, total_start_time)

    # 6. 검증 성능 평가
    print_progress("📈 검증 성능 평가 중...", total_start_time)
    X_val = val_data[feature_cols]
    y_val = val_data['demand']
    val_pred = model.predict(X_val)
    val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
    val_r2 = r2_score(y_val, val_pred)
    print_progress(f"📊 검증 성능 - RMSE: {val_rmse:.4f}, R²: {val_r2:.4f}", total_start_time)

    # 7. 2022년 검증 시각화
    create_xgboost_validation_visualization(val_data, val_pred, total_start_time)

    # 8. 미래 예측 (2023-2024)
    print_progress("🔮 미래 예측 중...", total_start_time)
    # 배치 예측 버전으로 교체
    result_df = predict_future_xgboost_batched(model, demand_data, feature_cols, label_encoders, total_start_time, events_df=events_df)

    # 9. 결과 저장
    output_path = DATA_DIR / "xgboost_forecast_submission.csv"
    result_df.to_csv(output_path, index=False)
    print_progress(f"✅ XGBoost 모델 예측 완료: {time.time() - total_start_time:.1f}초", total_start_time)
    print_progress(f"📁 결과 저장: {output_path}", total_start_time)
    print_progress(f"📊 총 예측 수: {len(result_df):,}", total_start_time)
    print_progress(f"📈 평균 수요: {result_df['mean'].mean():.1f}", total_start_time)
    print_progress(f"📊 검증 RMSE: {val_rmse:.4f}", total_start_time)

    # 10. 전체 타임라인 시각화
    print_progress("📊 2018-2024 전체 시계열 예측 시각화 생성 중...", total_start_time)
    create_xgboost_full_timeline_visualization(demand_data, result_df, total_start_time)

    return result_df


def main():
    print_progress("=== XGBoost 모델 기반 고급 시계열 예측 ===")
    _ = generate_xgboost_forecast()
    print_progress("✅ XGBoost 모델 완료!")


if __name__ == "__main__":
    main()

