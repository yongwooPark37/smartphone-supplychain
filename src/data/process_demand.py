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

from typing import Dict
from pathlib import Path
import pandas as pd
import numpy as np

# === 1. soft event score 탐지 ===
def detect_soft_event_score(
    df: pd.DataFrame,
    sku_meta_path: Path,
    country_map: Dict[str, str],
    threshold_pct: float = 0.20,
    min_days: int = 10,
    max_days: int = 90
) -> pd.DataFrame:
    """
    1) SKU launch_date 기준으로 각 나라 세그먼트 분할
    2) 국가 단위로 일별 수요 집계
    3) 각 세그먼트마다 daily pct_change 계산 후
       threshold_pct 이상인 연속 구간을 ‘급등’으로 검출
    4) 구간 길이가 [min_days, max_days] 사이면
       누적 증가율(mag) × (length/max_days)로 event_score 부여
    """
    # 1) launch_date 로드 & 결합 → 판매 전 데이터 제거
    sku_meta = pd.read_csv(sku_meta_path, parse_dates=["launch_date"])
    temp = df.merge(sku_meta[["sku","launch_date"]], on="sku", how="left")
    temp = temp[temp["date"] >= temp["launch_date"]]
    temp["country"] = temp["city"].map(country_map)

    # 2) 국가·일별 수요 집계
    daily = (
        temp.groupby(["country","date"], as_index=False)["demand"]
            .sum()
            .sort_values(["country","date"])
    )

    # 3) country별 launch_dates 로 세그먼트 경계 생성
    launch_dates = {
        c: sorted(sku_meta.loc[sku_meta['sku'].isin(
                    temp.loc[temp['country']==c,'sku']), 'launch_date'].unique())
        for c in daily['country'].unique()
    }

    results = []
    # 4) 국가별로 처리
    for country, grp in daily.groupby("country"):
        grp = grp.reset_index(drop=True)
        dates = grp["date"]
        demands = grp["demand"].values

        # pct change
        pct = pd.Series(demands).pct_change().fillna(0).values

        # 세그먼트 경계: [start0=first_date] + launch_dates + [end=last_date+1]
        bounds = [grp.loc[0,"date"]] + launch_dates.get(country,[]) + [grp.loc[len(grp)-1,"date"] + pd.Timedelta(days=1)]

        for start, end in zip(bounds, bounds[1:]):
            mask = (grp["date"] >= start) & (grp["date"] < end)
            idxs = np.where(mask)[0]
            if len(idxs)==0: continue

            # 5) 연속 급등 구간 찾기
            i = idxs[0]
            while i <= idxs[-1]:
                if pct[i] >= threshold_pct:
                    j = i
                    while j+1 <= idxs[-1] and pct[j+1] >= threshold_pct:
                        j += 1
                    length = j - i + 1
                    if min_days <= length <= max_days:
                        # 누적 증가율
                        mag = (demands[j] - demands[i]) / demands[i]
                        score = mag * (length / max_days)
                        # 해당 구간에 점수 할당
                        for k in range(i, j+1):
                            results.append({
                                "country": country,
                                "date": grp.loc[k,"date"],
                                "event_score": score
                            })
                    i = j+1
                else:
                    i += 1

    # 6) DataFrame으로 정리, 0점 채우기
    res_df = (
        pd.DataFrame(results, columns=["country","date","event_score"])
          .drop_duplicates(["country","date"])
    )
    # 전체 날짜·국가 조합에 0점 채우기
    full = (
        daily[["country","date"]]
        .merge(res_df, on=["country","date"], how="left")
        .fillna({"event_score": 0})
    )
    return full.sort_values(["country","date"]).reset_index(drop=True)

def plot_event_score(event_score_df: pd.DataFrame, countries: List[str]=None):
    """
    Soft Event Score 시계열 시각화

    Parameters
    ----------
    event_score_df : pd.DataFrame
        ['date','country','event_score'] 컬럼 포함된 데이터프레임
    countries : List[str], optional
        플롯할 국가 코드 리스트. None이면 전체.
    """
    df = event_score_df.copy()
    df['date'] = pd.to_datetime(df['date'])
    if countries is None:
        countries = df['country'].unique()

    plt.figure(figsize=(14, 6))
    for country in countries:
        sub = df[df['country'] == country].sort_values('date')
        plt.plot(sub['date'], sub['event_score'], label=country)

    plt.title("Soft Event Score Over Time by Country")
    plt.xlabel("Date")
    plt.ylabel("Event Score")
    plt.legend(title="Country")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# === 2. 이상치 제거 ===
def remove_outliers_city_sku(
    df: pd.DataFrame,
    event_score_df: pd.DataFrame,
    sku_meta_path: Path,
    country_map: Dict[str, str],
    event_threshold: float = 0.15
) -> pd.DataFrame:
    sku_meta = pd.read_csv(sku_meta_path, parse_dates=["launch_date"])

    launches_by_country = {
        c: sorted(
            sku_meta.loc[sku_meta['sku'].isin(
                df.loc[df['city'].map(country_map)==c,'sku']
            ), 'launch_date'].unique()
        )
        for c in set(country_map.values())
    }

    df = df.copy()
    df["country"] = df["city"].map(country_map)
    df = df.merge(event_score_df, how="left", on=["date", "country"])
    df["event_score"] = df["event_score"].fillna(0)
    df["launch_event"] = (df["event_score"] > event_threshold).astype(int)

    cleaned_list = []
    for (city, sku), sub in df.groupby(['city','sku']):
        sub = sub.sort_values('date').set_index('date')
        country = country_map[city]
        sub['demand_cleaned'] = sub['demand'].astype(float)

        bounds = [sub.index.min()] + launches_by_country.get(country,[]) + [sub.index.max()]
        for start_b, end_b in zip(bounds, bounds[1:]):
            seg = sub.loc[start_b:end_b]
            if seg.empty: continue
            norm = seg.loc[seg['launch_event'] == 0, 'demand']
            if norm.empty: continue
            q1, q3 = norm.quantile([0.25, 0.75])
            iqr = q3 - q1
            upper = q3 + 5 * iqr
            out_idx = norm[norm > upper].index
            med = seg['demand'].rolling(7, center=True, min_periods=1).median()
            sub.loc[out_idx, 'demand_cleaned'] = med.loc[out_idx]

        tmp = sub.reset_index()[['date', 'city', 'sku', 'demand', 'demand_cleaned']]
        cleaned_list.append(tmp)

    result = pd.concat(cleaned_list, ignore_index=True)
    return result

# === 3. 시각화 ===
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

# === 4. 실행 ===
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

    with sqlite3.connect(input_db) as conn:
        df = pd.read_sql("SELECT * FROM demand_train", conn, parse_dates=['date'])

    event_score_df = detect_soft_event_score(
        df,
        sku_meta_path=sku_meta_path,
        country_map=country_map,
        threshold_pct = 0.20,
        min_days = 10,
        max_days = 90
    )
    plot_event_score(event_score_df, countries=['USA','KOR','DEU','FRA','JPN'])

    cleaned_df = remove_outliers_city_sku(
        df,
        event_score_df,
        sku_meta_path=sku_meta_path,
        country_map=country_map,
        event_threshold=0.15
    )

    plot_sample(cleaned_df, sample_n=5)

    with sqlite3.connect(output_db) as conn:
        final_df = cleaned_df[['date','city','sku','demand_cleaned']].rename(columns={'demand_cleaned':'demand'})
        final_df.to_sql("demand_train", conn, index=False, if_exists='replace')

main()
