# src/models/exact_baseline_model.py
# 베이스라인과 완전히 동일한 모델

import pandas as pd
from sqlalchemy import create_engine
from datetime import datetime
from sklearn.linear_model import LinearRegression
import numpy as np
from pathlib import Path

# 경로 설정
SCRIPT = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT.parents[2]
DATA_DIR = PROJECT_ROOT / "data"

# 1. DB 연결 및 로드
engine = create_engine(f"sqlite:///{DATA_DIR}/demand_train_processed.db")
df = pd.read_sql("SELECT * FROM demand_train", con=engine)
df["date"] = pd.to_datetime(df["date"])

# 2. 학습 데이터 필터링 (2018~2022)
train_df = df[(df["date"] >= "2018-01-01") & (df["date"] <= "2022-12-31")]

# 3. 예측 대상 날짜 생성
future_dates = pd.date_range(start="2023-01-01", end="2024-12-31", freq="D")
future_days = np.array([d.toordinal() for d in future_dates]).reshape(-1, 1)

# 4. 결과 저장 리스트
results = []

# 5. 그룹별 회귀 학습 및 예측
grouped = train_df.groupby(["city", "sku"])

for (city, sku), group in grouped:
    group = group.sort_values("date")
    ts = group[["date", "demand"]].copy()
    ts = ts.dropna()

    if len(ts) < 10:
        continue  # 데이터 부족 시 스킵

    # 날짜를 수치형으로 변환 (ordinal)
    ts["x"] = ts["date"].map(datetime.toordinal)
    X_train = ts["x"].values.reshape(-1, 1)
    y_train = ts["demand"].values

    # 회귀 모델 학습
    model = LinearRegression()
    model.fit(X_train, y_train)

    # 미래 예측
    y_pred = model.predict(future_days)

    # 결과 저장
    for date, pred in zip(future_dates, y_pred):
        results.append({
            "date": date.strftime("%Y-%m-%d"),
            "sku": sku,
            "city": city,
            "mean": round(max(pred, 0), 2)  # 음수 방지
        })

# 6. 저장
df_out = pd.DataFrame(results)
output_path = DATA_DIR / "exact_baseline_forecast.csv"
df_out.to_csv(output_path, index=False)
print(f"✅ 저장 완료: {output_path} ({df_out.shape[0]:,} rows)")

# 결과 요약
print(f"\n=== Exact Baseline Forecast Summary ===")
print(f"Total predictions: {len(df_out):,}")
print(f"Date range: {df_out['date'].min()} to {df_out['date'].max()}")
print(f"Average daily demand: {df_out['mean'].mean():.1f}")
print(f"Max daily demand: {df_out['mean'].max():,}")
print(f"Min daily demand: {df_out['mean'].min()}") 