# src/models/xgboost_subset_berlin_sku0001.py
# XGBoost 예측 파이프라인(2022 검증 동일) + 2023-01-01 ~ 2023-06-30 구간에 대해
# Berlin, SKU0001 조합만 배치 예측으로 수행하는 서브셋 실행 스크립트

import pandas as pd
import numpy as np
from datetime import datetime
import time
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
from pathlib import Path

# 패키지 설치 없이 스크립트 단독 실행을 지원하기 위한 경로 주입
_SCRIPT = Path(__file__).resolve()
_PROJECT_ROOT = _SCRIPT.parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# 기존 xgboost_model의 유틸/전처리/학습/시각화 재사용
from src.models.xgboost_model import (
    print_progress,
    load_enhanced_training_data,
    prepare_lightgbm_features,
    analyze_multicollinearity,
    train_xgboost_model,
    create_xgboost_validation_visualization,
    create_xgboost_full_timeline_visualization,
    get_hardcoded_event_periods,
    DATA_DIR,
)

def create_subset_visualization(demand_data: pd.DataFrame, result_df: pd.DataFrame, start_time):
    """Berlin - SKU0001 전용 타임라인 시각화 (2022-2024, 예측은 2023-01~06만 표시)"""
    city, sku = 'Berlin', 'SKU0001'
    print_progress(f"📊 Plotting {city}-{sku}...", start_time)

    # 실제 데이터 (2022-2022)
    actual = demand_data[(demand_data['city'] == city) & (demand_data['sku'] == sku)][['date','demand']].copy()
    actual = actual[actual['date'] >= pd.Timestamp('2022-01-01')]
    # 예측 데이터 (서브셋)
    pred = result_df[(result_df['city'] == city) & (result_df['sku'] == sku)][['date','mean']].copy()
    pred = pred.rename(columns={'mean':'demand'})

    plt.figure(figsize=(14,5))
    if len(actual) > 0:
        actual = actual.sort_values('date')
        plt.plot(actual['date'], actual['demand'], label='Actual (2018-2022)', color='blue', linewidth=2)
    if len(pred) > 0:
        pred = pred.sort_values('date')
        plt.plot(pred['date'], pred['demand'], label='Predicted (2023-01~06)', color='red', linewidth=2)

    plt.axvline(pd.Timestamp('2023-01-01'), color='green', linestyle=':', alpha=0.7, label='Forecast start')
    plt.title('XGBoost Subset: Berlin - SKU0001 (2022-2024, Pred 2023-01~06)')
    plt.xlabel('Date')
    plt.ylabel('Demand')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    # x축 범위 고정 (오른쪽 끝: 2023-06-30)
    plt.xlim([pd.Timestamp('2022-06-01'), pd.Timestamp('2023-06-30')])

    out_path = DATA_DIR / 'xgb_subset_berlin_sku0001_timeline.png'
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print_progress(f"📁 Saved subset plot: {out_path}", start_time)
    
def predict_future_xgboost_subset(model, demand_data, feature_cols, label_encoders, start_time):
    """Berlin, SKU0001만 2023-01-01 ~ 2023-06-30 기간 예측 (날짜 배치 예측)"""
    print_progress("🔮 XGBoost 서브셋 미래 예측 (Berlin-SKU0001, 2023-01~06) 시작", start_time)

    # 2023-06-30까지만
    future_dates = pd.date_range(start='2023-01-01', end='2023-06-30', freq='D')

    # 외생/보조 피처 로드 (xgboost_model과 동일 처리)
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

    cov_future = country_cov.merge(cov_date, on='date', how='left').sort_values(['country','date'])
    # 간단 보간/보정 (서브셋에서도 동일)
    try:
        cov_future[['confidence_index','spend_usd','avg_temp','humidity']] = (
            cov_future.groupby('country')[['confidence_index','spend_usd','avg_temp','humidity']].ffill()
        )
        for c in ['confidence_index','spend_usd','avg_temp','humidity']:
            med = cov_future[c].median() if cov_future[c].notna().any() else 0.0
            cov_future[c] = cov_future[c].fillna(med)
        if 'season' in cov_future.columns:
            cov_future['season'] = cov_future.groupby('country')['season'].ffill().fillna('')
        cov_date = cov_date.sort_values('date')
        for c in ['brent_usd','pct_change','volatility_7d'] + fx_cols:
            if c in cov_date.columns:
                cov_date[c] = cov_date[c].ffill()
    except Exception as e:
        print(f"[WARN] cov ffill failed: {e}")

    # 조합 고정: Berlin, SKU0001
    city_target = 'Berlin'
    sku_target = 'SKU0001'
    combos = pd.DataFrame([{'city': city_target, 'sku': sku_target}])

    # 최근 데이터/버퍼 준비
    result = []
    cov_future_idx = cov_future.set_index(['date','country']).sort_index()
    cov_date_idx = cov_date.set_index('date').sort_index()
    sku_launch_map = {row['sku']: row['launch_date'] for _, row in sku_meta.iterrows()}

    # 해당 조합의 최근 60개
    recent_data = demand_data[(demand_data['city'] == city_target) & (demand_data['sku'] == sku_target)].tail(60)
    if len(recent_data) == 0:
        print("⚠️ Berlin-SKU0001 데이터가 없습니다.")
        return pd.DataFrame(columns=['sku','city','date','mean'])

    buf = recent_data['demand'].dropna().tolist() or [0.0]
    country = recent_data['country'].iloc[0]
    family = recent_data['family'].iloc[0]
    season_fixed = recent_data['season'].iloc[0]

    # 바닥값(최근 14일 평균의 10%)
    floor_val = max(0.0, float(recent_data['demand'].tail(14).mean()) * 0.10)

    print_progress(f"📊 예측 대상(서브셋): {city_target}-{sku_target} × {len(future_dates)}일", start_time)

    for date in tqdm(future_dates, desc="예측 진행(서브셋)"):
        row = {
            'date': date, 'city': city_target, 'sku': sku_target,
            'month': date.month, 'dayofyear': date.dayofyear, 'weekday': date.weekday(),
            'country': country, 'family': family, 'season': season_fixed,
        }

        # 국가 공변량
        try:
            cr = cov_future_idx.loc[(date, country)]
            row['confidence_index'] = cr.get('confidence_index', 0)
            row['spend_usd'] = cr.get('spend_usd', 0)
            row['avg_temp'] = cr.get('avg_temp', 0)
            row['humidity'] = cr.get('humidity', 0)
            row['season'] = cr.get('season', row.get('season',''))
        except Exception:
            for c in ['confidence_index','spend_usd','avg_temp','humidity']:
                row[c] = 0

        # 날짜 공변량
        try:
            dr = cov_date_idx.loc[date]
            row['brent_usd'] = dr.get('brent_usd', 0)
            row['pct_change'] = dr.get('pct_change', 0)
            row['volatility_7d'] = dr.get('volatility_7d', 0)
            for fx in fx_cols:
                row[fx] = dr.get(fx, 0)
        except Exception:
            for c in ['brent_usd','pct_change','volatility_7d'] + fx_cols:
                row[c] = 0

        # days_since_launch
        ld = sku_launch_map.get(sku_target, None)
        if ld is not None and pd.notna(ld):
            row['days_since_launch'] = max(0, (date - ld).days)
        else:
            row['days_since_launch'] = 0

        # 이벤트 플래그
        is_event = 0
        for (cty, year), (s, e) in get_hardcoded_event_periods().items():
            if cty == country and date >= pd.to_datetime(s) and date <= pd.to_datetime(e):
                is_event = 1
                break
        row['is_event'] = is_event

        # 수요 시계열 피처
        for lag in [1,3,7,14]:
            row[f'demand_lag_{lag}'] = buf[-lag] if len(buf) >= lag else (buf[0] if len(buf) > 0 else 0)
        for window in [7,14]:
            if len(buf) > 0:
                series = buf[-window:] if len(buf) >= window else buf
                row[f'demand_rolling_mean_{window}'] = float(np.mean(series))
                row[f'demand_rolling_std_{window}'] = float(np.std(series, ddof=1)) if len(series) > 1 else 0.0
            else:
                row[f'demand_rolling_mean_{window}'] = 0.0
                row[f'demand_rolling_std_{window}'] = 0.0

        # 가격/할인 최근값 및 시계열
        if 'unit_price' in recent_data.columns and len(recent_data) > 0:
            row['unit_price'] = float(recent_data['unit_price'].ffill().iloc[-1])
        if 'discount_pct' in recent_data.columns and len(recent_data) > 0:
            row['discount_pct'] = float(recent_data['discount_pct'].ffill().iloc[-1])
            if row['discount_pct'] > 1.0:
                row['discount_pct'] = row['discount_pct'] / 100.0
        for col in ['discount_pct','unit_price']:
            if col in recent_data.columns:
                for lag in [1,3,7,14]:
                    row[f'{col}_lag_{lag}'] = recent_data[col].iloc[-lag] if len(recent_data) >= lag else (recent_data[col].iloc[0] if len(recent_data) > 0 else 0)
                for window in [7,14]:
                    vals = recent_data[col].tail(window)
                    row[f'{col}_rolling_mean_{window}'] = vals.mean() if len(vals) > 0 else (recent_data[col].mean() if len(recent_data) > 0 else 0)

        # 카테고리 인코딩
        for col in ['city','sku','country','family','season']:
            enc_col = f"{col}_encoded"
            if enc_col in feature_cols:
                val = str(row.get(col,''))
                if col in label_encoders and len(label_encoders[col].classes_) > 0:
                    if val not in label_encoders[col].classes_:
                        val = str(label_encoders[col].classes_[0])
                    row[enc_col] = label_encoders[col].transform([val])[0]
                else:
                    row[enc_col] = 0

        # 단일 행 예측
        row_df = pd.DataFrame([row])
        for c in feature_cols:
            if c not in row_df.columns:
                row_df[c] = 0
        pred = float(model.predict(row_df[feature_cols])[0])
        # 바닥값 적용
        pred = max(pred, floor_val)
        pred = int(max(0, round(pred)))

        # 결과 + 버퍼 갱신
        result.append({'sku': sku_target, 'city': city_target, 'date': date, 'mean': pred})
        buf.append(pred)
        if len(buf) > 60:
            buf = buf[-60:]

    return pd.DataFrame(result)[['sku','city','date','mean']]


def generate_xgboost_forecast_subset():
    print_progress("=== XGBoost 서브셋(베를린-SKU0001, 2023-01~06) 예측 생성 ===")
    total_start_time = time.time()

    # 1) 데이터 로드
    demand_data, events_df, label_encoders = load_enhanced_training_data()

    # 2) 피처 준비
    demand_data, feature_cols = prepare_lightgbm_features(demand_data)

    # 3) 분할
    print_progress("📊 데이터 분할 중...", total_start_time)
    train_data = demand_data[demand_data['date'] < '2022-01-01'].copy()
    val_data = demand_data[(demand_data['date'] >= '2022-01-01') & (demand_data['date'] < '2023-01-01')].copy()

    # 4) 다중공선성 분석(샘플)
    print_progress("🔍 다중공선성 분석 중...", total_start_time)
    X_train_sample = train_data[feature_cols].sample(n=min(10000, len(train_data)), random_state=42)
    _ = analyze_multicollinearity(X_train_sample, feature_cols, total_start_time)

    # 5) 학습
    model = train_xgboost_model(train_data, val_data, feature_cols, total_start_time)

    # 6) 검증
    print_progress("📈 검증 성능 평가 중...", total_start_time)
    X_val = val_data[feature_cols]
    y_val = val_data['demand']
    val_pred = model.predict(X_val)
    create_xgboost_validation_visualization(val_data, val_pred, total_start_time)

    # 7) 서브셋 미래 예측
    print_progress("🔮 미래 예측(서브셋) 중...", total_start_time)
    result_df = predict_future_xgboost_subset(model, demand_data, feature_cols, label_encoders, total_start_time)

    # 8) 저장
    output_path = DATA_DIR / "xgboost_forecast_submission_subset_berlin_sku0001.csv"
    result_df.to_csv(output_path, index=False)
    print_progress(f"✅ 서브셋 예측 완료. 저장: {output_path}", total_start_time)

    # 9) Berlin-SKU0001 전용 시각화
    print_progress("📊 Berlin-SKU0001 전용 시각화 생성 중...", total_start_time)
    create_subset_visualization(demand_data, result_df, total_start_time)

    return result_df


def main():
    print_progress("=== XGBoost 서브셋 실행 시작 ===")
    _ = generate_xgboost_forecast_subset()
    print_progress("✅ XGBoost 서브셋 실행 완료")


if __name__ == "__main__":
    main()




