import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple
from itertools import groupby
from operator import itemgetter

# === 경로 설정 ===
SCRIPT = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT.parents[2]
DATA_DIR = PROJECT_ROOT / "data"

def detect_launch_events(
    df: pd.DataFrame,
    sku_meta_path: Path,
    country_map: Dict[str, str],
    threshold: float = 1.5,
    window: int = 15
) -> Dict[str, List[Tuple[pd.Timestamp, pd.Timestamp]]]:
    """
    - SKU 출시 전 데이터 제외
    - 년도별, 국가별로 데이터 분할하여 이벤트 탐지
    - 매년 하나의 기간만 AUC 기준으로 선택
    - length 30~90일 우선 필터, 없으면 AUC 최대 구간 선택
    """
    import pandas as pd
    from itertools import groupby

    # SKU 출시 전 제거 + 매핑 + 일별 합산
    sku_meta = pd.read_csv(sku_meta_path, parse_dates=["launch_date"])
    temp = df.merge(sku_meta[["sku","launch_date"]], on="sku", how="left")
    temp = temp[temp["date"] >= temp["launch_date"]]
    temp["country"] = temp["city"].map(country_map)
    daily = temp.groupby(["date","country"], as_index=False)["demand"].sum()

    candidates = []
    # 년도별로 순회
    years = daily['date'].dt.year.unique()
    for year in years:
        yearly = daily[daily['date'].dt.year == year]
        for country, sub in yearly.groupby('country'):
            sub = sub.sort_values('date').reset_index(drop=True)
            sub['rolling_avg']   = sub['demand'].rolling(window, min_periods=1).mean()
            sub['long_term_avg'] = sub['demand'].expanding(min_periods=window*2).mean()
            sub['ratio']         = sub['rolling_avg'] / sub['long_term_avg']
            sub['is_event_day']  = (sub['ratio'] > threshold).astype(int)
            dates = sub.loc[sub['is_event_day']==1, 'date']
            # 연속 구간 파악
            for _, grp in groupby(enumerate(dates), lambda ix: ix[0]-ix[1].toordinal()):
                days = [d for _,d in grp]
                s, e = min(days), max(days)
                length = (e - s).days + 1
                # AUC 계산
                auc = sub.set_index('date').loc[s:e]['ratio'].sub(1).clip(lower=0).sum()
                candidates.append({
                    'year': year,
                    'country': country,
                    'start': s,
                    'end': e,
                    'length': length,
                    'auc': auc
                })
    df_cand = pd.DataFrame(candidates)
    events: Dict[str,List[Tuple[pd.Timestamp,pd.Timestamp]]] = {}
    if df_cand.empty:
        return events
    for yr, grp in df_cand.groupby('year'):
        # 30~90일 후보
        valid = grp[(grp['length'] >= 30) & (grp['length'] <= 90)]
        if not valid.empty:
            best = valid.loc[valid['auc'].idxmax()]
        else:
            # 유효 구간 없다면 AUC 최대 구간
            best = grp.loc[grp['auc'].idxmax()]
        events.setdefault(best['country'], []).append((best['start'], best['end']))
    return events

# === 2. 이상치 제거 ===
def remove_outliers_city_sku(
    df: pd.DataFrame,
    launch_event_ranges: Dict[str, List[Tuple[pd.Timestamp, pd.Timestamp]]],
    sku_meta_path: Path
) -> pd.DataFrame:
    """
    - launch_event 기간 제외
    - SKU 출시일 기준 구간 분할
    - 각 구간 내 IQR 기반 이상치 탐지 및 7일 중앙값 대체
    """
    sku_meta = pd.read_csv(sku_meta_path, parse_dates=["launch_date"])

    country_map = {
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
    # 국가별 출시일 경계
    launches_by_country = {
        c: sorted(
            sku_meta.loc[sku_meta['sku'].isin(
                df.loc[df['city'].map(country_map)==c,'sku']
            ), 'launch_date'].unique()
        )
        for c in set(country_map.values())
    }

    cleaned_list = []
    for (city, sku), sub in df.groupby(['city','sku']):
        sub = sub.sort_values('date').set_index('date')
        country = country_map[city]
        sub['launch_event'] = 0
        for start,end in launch_event_ranges.get(country,[]):
            sub.loc[start:end,'launch_event'] = 1
        sub['demand_cleaned'] = sub['demand'].astype(float)

        bounds = [sub.index.min()] + launches_by_country.get(country,[]) + [sub.index.max()]
        for start_b,end_b in zip(bounds,bounds[1:]):
            seg = sub.loc[start_b:end_b]
            if seg.empty: continue
            norm = seg.loc[seg['launch_event']==0,'demand']
            if norm.empty: continue
            q1,q3 = norm.quantile([0.25,0.75])
            iqr = q3-q1
            upper = q3+5*iqr
            out_idx = norm[norm>upper].index
            med = seg['demand'].rolling(7,center=True,min_periods=1).median()
            sub.loc[out_idx,'demand_cleaned'] = med.loc[out_idx]

        tmp = sub.reset_index()[['date','city','sku','demand','demand_cleaned']]
        cleaned_list.append(tmp)

    result = pd.concat(cleaned_list, ignore_index=True)
    return result

# === 3. 시각화 샘플 ===
def plot_sample(df: pd.DataFrame, sample_n: int = 5):
    sample_keys = df[['city', 'sku']].drop_duplicates().sample(sample_n, random_state=41).values
    for city, sku in sample_keys:
        subset = df[(df['city'] == city) & (df['sku'] == sku)].sort_values('date')
        plt.figure(figsize=(10, 4))
        plt.plot(subset['date'], subset['demand'], label='Original', alpha=0.5)
        plt.plot(subset['date'], subset['demand_cleaned'], label='Cleaned', linewidth=2)
        plt.title(f"City: {city}, SKU: {sku}")
        plt.legend()
        plt.tight_layout()
        plt.show()


# === 4. 메인 실행 ===
def main():
    input_db  = DATA_DIR / "demand_train.db"
    output_db = DATA_DIR / "demand_train_processed.db"
    sku_meta_path = DATA_DIR / "sku_meta.csv"
    country_map = {
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
    # 1) 원본 demand 로드
    with sqlite3.connect(input_db) as conn:
        df = pd.read_sql("SELECT * FROM demand_train", conn, parse_dates=['date'])

    # 2) 이벤트 탐지 (SKU 출시일 필터 포함)
    launch_events = detect_launch_events(
        df,
        sku_meta_path=sku_meta_path,
        country_map=country_map,
        window=15
    )
    print("Detected Launch Events:")
    for c, periods in launch_events.items():
        for s,e in periods:
            print(f"- {c}: {s.date()} ~ {e.date()}")

    # 3) 이상치 제거 & 시각화 샘플
    cleaned_df = remove_outliers_city_sku(
        df,
        launch_events,
        sku_meta_path=sku_meta_path
    )
    plot_sample(cleaned_df, sample_n=5)

    # 4) 처리된 수요 저장
    with sqlite3.connect(output_db) as conn:
        final_df = cleaned_df[['date','city','sku','demand_cleaned']].rename(columns={'demand_cleaned':'demand'})
        final_df.to_sql("demand_train", conn, index=False, if_exists='replace')

if __name__ == "__main__":
    main()
