!pip install --quiet gdown
import gdown

file_id = "1XjTLTFlvxQPmepHbzPACfLPoMs0SKf46"
url = f"https://drive.google.com/uc?id={file_id}"
output_zip = "downloaded_file.zip"

print("Downloading...")
gdown.download(url, output_zip, quiet=False)

import zipfile
import os

extract_dir = "extracted_contents"
os.makedirs(extract_dir, exist_ok=True)

with zipfile.ZipFile(output_zip, 'r') as zip_ref:
    zip_ref.extractall(extract_dir)

print(f"Done! '{extract_dir}'")

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

# 1) 데이터 로드 & 피벗
df = pd.read_csv("extracted_contents/data/consumer_confidence.csv", parse_dates=["month"])
wide = df.pivot(index="month", columns="country", values="confidence_index").sort_index()

# 2) 표준화 → PCA로 '글로벌 심리' 1~2개 요인 추출
scaler = StandardScaler()
Z = scaler.fit_transform(wide)              # shape: (T, 9)
pca = PCA(n_components=2).fit(Z)
global_factors = pca.transform(Z)           # 첫 PC = 공통 심리

# 3) Canada 후보 v1 : 첫 PC를 캐나다 스케일로 맞추기
#    (평균 100, 표준편차는 미국과 동일)
usa_mu, usa_sigma = wide["USA"].mean(), wide["USA"].std()
canada_v1 = global_factors[:, 0]            # 1차 요인
canada_v1 = (canada_v1 - canada_v1.mean()) / canada_v1.std()
canada_v1 = canada_v1 * usa_sigma + 100

# 4) Canada 후보 v2 : 교차검증 회귀
X = wide.drop(columns=["USA"])              # 예시: USA 예측용 실험
y = wide["USA"]
lags = 2                                     # t-1, t-2 추가
for lag in range(1, lags+1):
    X[f'PC1_lag{lag}'] = global_factors[:-lag, 0].tolist() + [None]*lag
X = X.dropna()
y = y.loc[X.index]

ridge = RidgeCV(alphas=[0.1, 1.0, 10.0]).fit(X, y)
print("RMSE(USA hold-out) :", np.sqrt(mean_squared_error(y, ridge.predict(X))))

# 캐나다는 열이 없으므로 결측으로 채우고 예측
full_X = wide.drop(columns=["USA"]).copy()
for lag in range(1, lags+1):
    full_X[f'PC1_lag{lag}'] = global_factors[:-lag, 0].tolist() + [None]*lag
canada_v2 = ridge.predict(full_X.dropna())

# 5) 최종 합치기 & 저장
canada_series = pd.Series(canada_v1, index=wide.index, name="CAN").round(5)

result_wide  = pd.concat([wide, canada_series], axis=1)
result_wide.columns.name = "country"

# 6) wide → long 형태로 되돌리기
result_long = (
    result_wide.round(5)          # 필요한 경우 반올림
               .stack()           # (month, country) → value
               .rename("confidence_index")
               .reset_index()     # 컬럼: ['month', 'country', 'confidence_index']
)

result_long["month"] = result_long["month"].dt.to_period("M").astype(str)
result_long.sort_values(["country", "month"], inplace=True)

# 7) 저장 (열 순서 그대로)
result_long.to_csv(
    "extracted_contents/data/consumer_confidence_with_canada_proxy.csv",
    index=False,
    columns=["month", "country", "confidence_index"]  # 열 순서 고정
)

import pandas as pd
import numpy as np

def add_local_fx_iso(df: pd.DataFrame) -> pd.DataFrame:
    """
    country 컬럼(ISO 3자리 코드)에 따라 local_fx 컬럼에
    해당 국가 환율만 채워 줌.
    USA -> 1.0, 나머지는 매핑된 'XXX=X' 컬럼 값. 
    매핑 없으면 NaN.
    """
    # ISO 코드 → 환율 컬럼 매핑
    iso_to_fx = {
        'USA': 'USD',        # USD: 1.0
        'DEU': 'EUR=X',     # 유로존 (독일, 프랑스 등)
        'FRA': 'EUR=X',
        'KOR': 'KRW=X',
        'JPN': 'JPY=X',
        'GBR': 'GBP=X',
        'CAN': 'CAD=X',
        'AUS': 'AUD=X',
        'BRA': 'BRL=X',
        'ZAF': 'ZAR=X'
    }

    df = df.copy()
    def _pick_fx(row):
        fx_col = iso_to_fx.get(row['country'])

        if fx_col is None:
            raise ValueError(f"No FX column found for country: {row['country']}")
            
        if fx_col == 'USD':
            # USA 이거나 매핑이 없는 경우
            return 1.0
        # 환율 컬럼이 실제로 있으면 그 값, 없으면 NaN
        return row.get(fx_col, np.nan)

    df['local_fx'] = df.apply(_pick_fx, axis=1)
    return df

import os
import gc
import pandas as pd
import sqlite3
import polars as pl
# ── 설정 ───────────────────────────────────────────────
DATA_DIR = 'extracted_contents/data'   # 실제 데이터 위치로 수정
OUT_DIR  = 'data'
os.makedirs(OUT_DIR, exist_ok=True)

# ── 1) 날짜·국가별 피처 전체 생성 (labour_policy 제외) ─────
cal  = pd.read_csv(os.path.join(DATA_DIR,'calendar.csv'),       parse_dates=['date'])
hol  = pd.read_csv(os.path.join(DATA_DIR,'holiday_lookup.csv'), parse_dates=['date'])
hol['is_holiday'] = 1

# --- weather: delta_humidity 보정 로직 추가
wth  = pd.read_csv(os.path.join(DATA_DIR,'weather.csv'),        parse_dates=['date'])
# country, date 순으로 정렬 후 humidity 차분으로 delta_humidity 계산
wth = wth.sort_values(['country','date'])
wth['delta_humidity'] = wth.groupby('country')['humidity'].diff()
# 2018-01-01은 0으로 고정
wth.loc[wth['date'] == pd.Timestamp('2018-01-01'), 'delta_humidity'] = 0

# --- oil_price: 주말은 마지막 영업일 기준으로 forward-fill
oil  = pd.read_csv(os.path.join(DATA_DIR,'oil_price.csv'),      parse_dates=['date'])
oil = (
    oil
    .set_index('date')
    .resample('D')     # 모든 날짜로 확장
    .ffill()           # 직전 값으로 채우기
    .reset_index()
)

# --- currency: 마찬가지로 주말은 마지막 영업일 값으로 채우기
cur  = pd.read_csv(os.path.join(DATA_DIR,'currency.csv'),       parse_dates=['Date'])
cur = (
    cur
    .rename(columns={'Date':'date'})
    .set_index('date')
    .resample('D')
    .ffill()
    .reset_index()
)


cons = pd.read_csv(os.path.join(DATA_DIR,'consumer_confidence_with_canada_proxy.csv'),
                   parse_dates=['month'])
cons['month'] = cons['month'].dt.to_period('M')


df = (
    cal
    .merge(hol[['date','country','is_holiday']], on=['date','country'], how='left')
    .fillna({'is_holiday':0})
    .merge(wth, on=['date','country'], how='left')
    .merge(oil, on='date', how='left')
    .merge(cur, left_on='date', right_on='date', how='left')
    .assign(month=lambda d: d['date'].dt.to_period('M'))
    .merge(cons[['month','country','confidence_index']],
           on=['month','country'], how='left')
    .drop(columns=['month'])
)
df.rename(columns={'confidence_index':'CAN=X'}, inplace=True)
date_feat = df.copy()
del df, cal, hol, wth, oil, cur, cons
gc.collect()

# ── 2) SKU 메타 추출 ─────────────────────────────────
sku_meta = pd.read_csv(os.path.join(DATA_DIR,'sku_meta.csv'),
                       parse_dates=['launch_date'])
sku_meta['launch_date'] = pd.to_datetime(sku_meta['launch_date'])
needed_sku = ['family','storage_gb','colour','life_days','launch_date']
sku_map = sku_meta.set_index('sku')[needed_sku].to_dict()
del sku_meta
gc.collect()

# ── 3) price_promo_train 로드 & imputation 자료 생성 ────
ppt = pd.read_csv(os.path.join(DATA_DIR,'price_promo_train.csv'),
                  parse_dates=['date'])
# 3-1) Base price: 2022년 평균
base_price = (ppt[ppt['date'].dt.year == 2022]
              .groupby(['sku','city'])['unit_price']
              .mean())
base_discount = (ppt[ppt['date'].dt.year == 2022]
                 .groupby(['sku','city'])['discount_pct']
                 .mean())
base_price_map = base_price.to_dict()
base_discount_map = base_discount.to_dict()

# 3-2) Last-month forecast-forward
last_date = ppt['date'].max()
last_data = ppt[ppt['date'] == last_date][['sku','city','unit_price','discount_pct']]
last_price_map = dict(zip(zip(last_data['sku'], last_data['city']), last_data['unit_price']))
last_discount_map = dict(zip(zip(last_data['sku'], last_data['city']), last_data['discount_pct']))

# 3-3) Monthly average by month-of-year
ppt['month'] = ppt['date'].dt.month
monthly_avg = (ppt.groupby(['sku','city','month'])
               [['unit_price','discount_pct']]
               .mean()
               .rename(columns={'unit_price':'unit_price_month',
                                'discount_pct':'discount_pct_month'}))
monthly_price_map = monthly_avg['unit_price_month'].to_dict()
monthly_discount_map = monthly_avg['discount_pct_month'].to_dict()
del ppt, base_price, base_discount, last_data, monthly_avg
gc.collect()

# ── 4) 학습용 데이터셋 생성 (2018–2022) ─────────────────
conn = sqlite3.connect(os.path.join(DATA_DIR,'demand_train.db'))
demand = pd.read_sql('SELECT * FROM demand_train', conn, parse_dates=['date'])
site_cand = pd.read_csv(os.path.join(DATA_DIR,'site_candidates.csv'))
city_country = site_cand[['city','country']].drop_duplicates()
demand = demand.merge(city_country, on='city', how='left')

marketing = pd.read_csv(os.path.join(DATA_DIR,'marketing_spend.csv'),
                        parse_dates=['date'])

train = (
    demand
    .merge(date_feat,   on=['date','country'],    how='left')
    .merge(marketing,   on=['date','country'],    how='left')
)
del demand
gc.collect()

for col in needed_sku:
    train[col] = train['sku'].map(sku_map[col])

# price_promo_train은 레이블 학습용으로만 사용
train = train.merge(
    pd.read_csv(os.path.join(DATA_DIR,'price_promo_train.csv'),
                parse_dates=['date']),
    on=['date','sku','city'], how='left'
)
train.drop(columns=["unit_price"], inplace=True)
ppt_full = pd.read_csv(os.path.join(DATA_DIR,'price_promo_train.csv'),
                       parse_dates=['date'])
# discount_pct == 0.0 인 원가격만 추출해 sku, city 별 평균 계산
orig_price_map = (
    ppt_full[ppt_full['discount_pct'] == 0.0]
    .groupby(['sku','city'])['unit_price']
    .mean()
    .to_dict()
)
train['unit_price'] = train.apply(
    lambda r: orig_price_map.get((r['sku'], r['city'])),
    axis=1
)


FX = ["CAD=X","AUD=X","BRL=X","ZAR=X","EUR=X","KRW=X","JPY=X","GBP=X","CAN=X"]
# ONEHOT_COLS = ["season","category","colour"]
# LABEL_COLS  = ["country","family"]
# DELETE_COLS = ['season_nan', 'category_nan', 'colour_nan']

# 1) 환율 forward-fill (벡터화)
train[FX] = train.groupby(["sku","city"])[FX].ffill()

# 2) days_since_launch
train["days_since_launch"] = (
    (train["date"] - train["launch_date"]).dt.days
).astype("float32")

train = train.drop(columns=["launch_date"])
train = add_local_fx_iso(train)
train = train.drop(columns=FX)
# 4) 인코딩
# train = pd.get_dummies(train, columns=ONEHOT_COLS, dummy_na=True, dtype="float32")
# train = train.drop(columns=DELETE_COLS)

# for col in LABEL_COLS:
#     train[col] = train[col].astype("category").cat.codes.astype("float32")


# train.to_csv(os.path.join(OUT_DIR,'train_master.csv'), index=False)
pl.from_pandas(train).write_csv(os.path.join(OUT_DIR,'train_master.csv'))
print("Saved train_master.csv")
gc.collect()

# ── 5) 예측용 데이터셋 생성 (2023–2024) ────────────────
forecast = pd.read_csv(os.path.join(DATA_DIR,'forecast_submission_template.csv'),
                       parse_dates=['date'])
forecast = forecast.merge(city_country, on='city', how='left')
test = (
    forecast
    .merge(date_feat, on=['date','country'], how='left')
    .merge(marketing, on=['date','country'], how='left')
)
del forecast, marketing
gc.collect()

# 필요한 SKU 피처 attach
for col in needed_sku:
    test[col] = test['sku'].map(sku_map[col])

# ── 5-1) original_price 추가: discount_pct == 0.0 인 시점의 unit_price 매핑 ──
ppt_full = pd.read_csv(os.path.join(DATA_DIR,'price_promo_train.csv'),
                       parse_dates=['date'])
# discount_pct == 0.0 인 원가격만 추출해 sku, city 별 평균 계산
orig_price_map = (
    ppt_full[ppt_full['discount_pct'] == 0.0]
    .groupby(['sku','city'])['unit_price']
    .mean()
    .to_dict()
)
# test 에 unit_price 컬럼 추가 할인율 반영 x
test['unit_price'] = test.apply(
    lambda r: orig_price_map.get((r['sku'], r['city'])),
    axis=1
)
del ppt_full, orig_price_map
gc.collect()

# 1) 환율 forward-fill (벡터화)
test[FX] = test.groupby(["sku","city"])[FX].ffill()
test = add_local_fx_iso(test)
test = test.drop(columns=FX)

# 2) days_since_launch
test["days_since_launch"] = (
    (test["date"] - test["launch_date"]).dt.days
).astype("float32")
test = test.drop(columns=["launch_date"])
# 4) 인코딩
# test = pd.get_dummies(test, columns=ONEHOT_COLS, dummy_na=True, dtype="float32")
# for col in LABEL_COLS:
#     test[col] = test[col].astype("category").cat.codes.astype("float32")
# test = test.drop(columns=DELETE_COLS)

# test.to_csv(os.path.join(OUT_DIR,'test_master.csv'), index=False)
pl.from_pandas(test).write_csv(os.path.join(OUT_DIR,'test_master.csv'))
print("Saved test_master.csv")

pl.from_pandas(pd.concat([train, test])).write_csv(os.path.join(OUT_DIR,'all_master.csv'))
print("Saved all_master.csv")

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates   # ← 추가
from tqdm import tqdm
import os


dataset = pd.read_csv("data/train_master.csv", parse_dates=['date'])

# 같은 날짜, 같은 나라 기준으로 demand 합산
aggregated_df = dataset.groupby(['date', 'country'], as_index=False)['demand'].sum()
aggregated_df.to_csv("aggregated_df.csv", index=False)

# 결과 확인
print(aggregated_df.head())