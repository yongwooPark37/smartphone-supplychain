import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import random
import torch
from datetime import datetime, timedelta
from typing import List, Tuple, Union, Dict
import time

from sklearn.compose import ColumnTransformer 
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from darts import TimeSeries
from darts.dataprocessing.transformers import StaticCovariatesTransformer
from darts.dataprocessing.transformers.scaler import Scaler
from darts.models import TiDEModel, NaiveMovingAverage, TFTModel
from darts.metrics import mae, mse, smape
from darts.utils.losses import MAELoss, MapeLoss, SmapeLoss
import darts

from pytorch_lightning.callbacks.early_stopping import EarlyStopping

import torch
import torch.nn.functional as F
import torchmetrics
from torch import nn

def set_global_seed(seed: int = 42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # CuDNN 연산을 deterministic하게 만들어 주지만, 약간 느려질 수 있음
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 시드 값 고정
set_global_seed()

def to_darts_time_series_group(
    dataset: pd.DataFrame,
    target: Union[List[str],str],
    time_col: str,
    group_cols: Union[List[str],str],
    static_cols: Union[List[str],str]=None,
    past_cols: Union[List[str],str]=None,
    future_cols: Union[List[str],str]=None,
    freq: str=None,
    encode_static_cov: bool=True,
)-> Tuple[List[TimeSeries], List[TimeSeries], List[TimeSeries], List[TimeSeries]]:

    series_raw = TimeSeries.from_group_dataframe(
    dataset,
    time_col    =   time_col,
    group_cols  =   group_cols,  # individual time series are extracted by grouping `df` by `group_cols`
    static_cols =   static_cols,  # also extract these additional columns as static covariates (without grouping)
    value_cols  =   target,  # optionally, specify the time varying columns
    n_jobs      =   -1,
    verbose     =   False,
    freq        =   freq,
    )

    if encode_static_cov:
        static_cov_transformer = StaticCovariatesTransformer()
        series_encoded = static_cov_transformer.fit_transform(series_raw)
    else: series_encoded = []

    if past_cols:
        past_cov = TimeSeries.from_group_dataframe(
            dataset,
            time_col    =   time_col,
            group_cols  =   group_cols,
            value_cols  =   past_cols,
            n_jobs      =   -1,
            verbose     =   False,
            freq        =   freq,
            )
    else: past_cov = []

    if future_cols:
        future_cov = TimeSeries.from_group_dataframe(
            dataset,
            time_col    =   time_col,
            group_cols  =   group_cols,
            value_cols  =   future_cols,
            n_jobs      =   -1,
            verbose     =   False,
            freq        =   freq,
            )
    else: future_cov = []

    return series_raw, series_encoded, past_cov, future_cov

def split_grouped_darts_time_series(
    series: List[TimeSeries],
    split_date: Union[str, pd.Timestamp],
    min_date: Union[str, pd.Timestamp]=None,
    max_date: Union[str, pd.Timestamp]=None,
) -> Tuple[List[TimeSeries], List[TimeSeries]]:

    if min_date:
       raw_series = series.copy()
       series = []
       for s in raw_series:
        try: series.append(s.split_before(pd.Timestamp(min_date)-timedelta(1))[1])
        except: series.append(s)

    if max_date:
       raw_series = series.copy()
       series = []
       for s in raw_series:
        try: series.append(s.split_before(pd.Timestamp(max_date))[0])
        except: series.append(s)

    split_0 = [s.split_before(pd.Timestamp(split_date))[0] for s in series]
    split_1 = [s.split_before(pd.Timestamp(split_date))[1] for s in series]
    return split_0, split_1

def eval_forecasts(
    pred_series: Union[List[TimeSeries], TimeSeries],
    test_series: Union[List[TimeSeries], TimeSeries],
    error_metric: darts.metrics,
    plot: bool=False
) -> List[float]:

    errors = error_metric(test_series, pred_series)
    print(errors)
    if plot:
        plt.figure()
        plt.hist(errors, bins=50)
        plt.ylabel("Count")
        plt.xlabel("Error")
        plt.title(f"Mean error: {np.mean(errors):.3f}")
        plt.show()
        plt.close()
    return errors

def fit_mixed_covariates_model(
    model_cls,
    common_model_args: dict,
    specific_model_args: dict,
    model_name: str,
    past_cov: Union[List[TimeSeries], TimeSeries],
    future_cov: Union[List[TimeSeries], TimeSeries],
    train_series: Union[List[TimeSeries], TimeSeries],
    val_series: Union[List[TimeSeries], TimeSeries]=None,
    max_samples_per_ts: int=None,
    save:bool=False,
    path:str="",
):

    # Declarare model
    model = model_cls(model_name=model_name,
                    **common_model_args,
                    **specific_model_args)

    # Train model
    model.fit(
                    # TRAIN ARGS ===================================
                    series                = train_series,
                    past_covariates       = past_cov,
                    future_covariates     = future_cov,
                    max_samples_per_ts    = max_samples_per_ts,
                    # VAL ARGS ======================================
                    val_series            = val_series,
                    val_past_covariates   = past_cov,
                    val_future_covariates = future_cov,
                )

    if save: model.save(path)

def backtesting(model, series, past_cov, future_cov, start_date, horizon, stride):
  historical_backtest = model.historical_forecasts(
    series, past_cov, future_cov,
    start=start_date,
    forecast_horizon=horizon,
    stride=stride,  # Predict every N months
    retrain=False,  # Keep the model fixed (no retraining)
    overlap_end=False,
    last_points_only=False
  )
  maes = model.backtest(series, historical_forecasts=historical_backtest, metric=mae)

  return np.mean(maes)

def process_predictions(
    preds: List[TimeSeries],
    series_raw: List[TimeSeries],
    group_cols: List[str]
) -> pd.DataFrame:

    list_df = [serie.pd_dataframe() for serie in preds]
    for i in range(len(list_df)):
      list_df[i]['Date'] = preds[i].time_index
      for j in range(len(group_cols)):
        list_df[i][group_cols[j]] = series_raw[i].static_covariates[group_cols[j]].values[0]
    processed_preds =  pd.concat(list_df, ignore_index=True)
    return processed_preds

def price_weighted_mae(predictions, targets, prices):
    """
    Compute the price-weighted Mean Absolute Error (MAE).

    :param predictions: A list or 1D NumPy array of predicted values.
    :param targets: A list or 1D NumPy array of actual (ground truth) values.
    :param prices: A list or 1D NumPy array of prices corresponding to the targets.
    :return: The price-weighted MAE as a float.
    """
    # Ensure inputs are NumPy arrays
    predictions = np.array(predictions, dtype=np.float32)
    targets = np.array(targets, dtype=np.float32)
    prices = np.array(prices, dtype=np.float32)

    # Compute absolute error
    error = np.abs(targets - predictions)

    # Compute price-weighted error
    weighted_error = error * prices

    # Compute and return the mean of the weighted error
    return np.mean(weighted_error)

def local_iqr_clip(series, window=30, q1=0.25, q3=0.75, m=2.5):
    roll_q1 = series.rolling(window, center=True).quantile(q1)
    roll_q3 = series.rolling(window, center=True).quantile(q3)
    iqr = roll_q3 - roll_q1
    upper = roll_q3 + m * iqr
    return series.clip(0, upper)

class MultiTaskLossModule(nn.Module):
    def __init__(self):
        super(MultiTaskLossModule, self).__init__()
        self.alpha = 1.0
        self.beta = 1.0
        self.gamma = 1.0

    def forward(self, y_pred, y_true):
        # 내부에 바로 구현
        reg_pred   = y_pred[..., :2]
        reg_true   = y_true[..., :2]
        cls_pred   = y_pred[..., 2]
        cls_true   = y_true[..., 2]

        loss_demand   = F.mse_loss(reg_pred[..., 0], reg_true[..., 0])
        loss_discount = F.mse_loss(reg_pred[..., 1], reg_true[..., 1])
        loss_cls      = F.binary_cross_entropy_with_logits(cls_pred, cls_true)

        return self.alpha*loss_demand + self.beta*loss_discount + self.gamma*loss_cls

TEST_DATE = pd.Timestamp('2023-01-01')
VAL_DATE_OUT = pd.Timestamp('2022-01-01')
VAL_DATE_IN = pd.Timestamp('2021-01-01')
# MIN_TRAIN_DATE = pd.Timestamp('2015-06-01')

dataset = pd.read_csv("data/all_master.csv", parse_dates=["date"])
dataset = dataset.sort_values(by=["city", "sku", "date"])

LABELS = {
    "JPN": [("2019-01-01", "2019-02-28")],
    "KOR": [("2018-01-15", "2018-04-15")],
    "USA": [("2020-01-01", "2020-04-30"),
            ("2021-03-01", "2021-05-31")]
}

dataset['is_event'] = 0

for country, periods in LABELS.items():
    for start, end in periods:
        start_dt = pd.to_datetime(start)
        end_dt   = pd.to_datetime(end)
        mask = (
            (dataset["country"] == country) &
            (dataset["date"] >= start_dt) &
            (dataset["date"] <= end_dt)   # end 포함
        )
        dataset.loc[mask, "is_event"] = 1

print(dataset[["country", "date", "is_event"]].query("is_event==1").head())

preprocess = 'none'
# preprocess = 'iqr'
dataset = pd.get_dummies(dataset, columns=['season'], prefix='season')
# dataset = dataset[dataset['days_since_launch'] > 0]

dataset['month'] = dataset['date'].dt.month
dataset['day']   = dataset['date'].dt.day 
dataset['year']  = dataset['date'].dt.year
dataset['day_of_week'] = dataset['date'].dt.dayofweek

dataset['time_index'] = (dataset['date'] - pd.Timestamp('2018-01-01')).dt.days
dataset['time_index'] = dataset['time_index'].astype(np.float32)


target_col = ['demand', 'discount_pct', 'is_event']
time_col = 'date'
group_cols = ['sku','city']

drop_cols = [
    # static
    # 'country', 
    'category', 'family', 'storage_gb', 'colour', 
    # numeric
    # 'season_Fall', 'season_Spring', 'season_Summer', 'season_Winter', 'is_holiday',
    'avg_temp', 'humidity', 'precip_mm',
    'rain_mm', 'snow_mm', 'snow_depth_cm', 'pressure_msl', 'cloud_cover',
    'wind_speed_avg', 'wind_speed_max', 'wind_gust_max', 'wind_dir_mode',
    'shortwave_rad_MJ', 'vpd', 'cdd18', 'delta_temp',
    'delta_humidity',
]
drop_cols = []
# past_cols = ['EMA_30', 'MA_30']
past_cols = []
future_cols = ['season_Fall', 'season_Spring', 'season_Summer', 'season_Winter', 'is_holiday','avg_temp', 'min_temp', 'max_temp', 'dewpoint', 'humidity', 'precip_mm',
       'rain_mm', 'snow_mm', 'snow_depth_cm', 'pressure_msl', 'cloud_cover',
       'wind_speed_avg', 'wind_speed_max', 'wind_gust_max', 'wind_dir_mode',
       'shortwave_rad_MJ', 'vpd', 'hdd18', 'cdd18', 'delta_temp',
       'delta_humidity', 'brent_usd', 'local_fx', 'spend_usd', 'days_since_launch',
       'month', 'day', "year", "day_of_week", "time_index"
]
static_cols = ['country', 'category', 'family', 'storage_gb', 'colour', 'unit_price', 'life_days']

dataset = dataset.drop(columns=drop_cols)
future_cols = [col for col in future_cols if col not in drop_cols]
static_cols = [col for col in static_cols if col not in drop_cols]


if preprocess == 'clip':
    print('clip')
    low, high = dataset['demand'].quantile([0.00,0.95])
    dataset['demand'] = dataset['demand'].clip(low, high)
elif preprocess == 'iqr':
    print('iqr')
    dataset['demand'] = local_iqr_clip(dataset['demand'])
dataset['demand'] = np.log1p(dataset['demand'])
dataset['discount_pct'] = dataset['discount_pct']/100

series_raw, series, past_cov, future_cov = to_darts_time_series_group(
    dataset=dataset,
    target=target_col,
    time_col=time_col,
    group_cols=group_cols,
    past_cols=past_cols,
    future_cols=future_cols,
    static_cols=static_cols,
    freq='D', # daily
    encode_static_cov=True, # so that the models can use the categorical variables (Agency & Product)
)

train_val, test = split_grouped_darts_time_series(
    series=series,
    split_date=TEST_DATE
)

train, _ = split_grouped_darts_time_series(
    series=train_val,
    split_date=VAL_DATE_OUT
)

_, val = split_grouped_darts_time_series(
    series=train_val,
    split_date=VAL_DATE_IN
)

early_stopping_args = {
    "monitor": "val_loss",
    "patience": 10,
    "min_delta": 1e-3,
    "mode": "min",
}

pl_trainer_kwargs = {
    "max_epochs": 100,
    "accelerator": "gpu", 
    "callbacks": [EarlyStopping(**early_stopping_args)],
    "enable_progress_bar":True
}

common_model_args = {
    "output_chunk_length": 7,
    "input_chunk_length": 84,
    "pl_trainer_kwargs": pl_trainer_kwargs,
    "save_checkpoints": True,  # checkpoint to retrieve the best performing model state,
    "force_reset": True,
    "batch_size": 512,
    "random_state": 42,
}

encoders = {
    "position": {"past": ["relative"], "future": ["relative"]},
    "transformer": Scaler(),
}

best_hp = {
 'optimizer_kwargs': {'lr':0.0001},
 'loss_fn': MultiTaskLossModule(),
 'use_layer_norm': True,
 'use_reversible_instance_norm': True,
 'add_encoders':encoders,
 }

past_cov = None if not past_cov else past_cov

start = time.time()
## COMMENT TO LOAD PRE-TRAINED MODEL
fit_mixed_covariates_model(
    model_cls = TiDEModel,
    common_model_args = common_model_args,
    specific_model_args = best_hp,
    model_name = 'TiDE_model',
    past_cov = past_cov,
    future_cov = future_cov,
    train_series = train,
    # train_series=train_val,
    val_series = val,
    # val_series=None,
)
time_tide = time.time() - start

from typing import List, Optional
import pandas as pd
import torch
from tqdm.auto import tqdm
from darts import TimeSeries
from IPython.display import clear_output

def extend_covariate(ts: TimeSeries, n: int) -> TimeSeries:
    """
    ts   : 원본 future_covariate TimeSeries
    n    : 예측 스텝 수
    return: 예측 구간까지 0 으로 채운 TimeSeries
    """
    freq       = ts.freq_str
    last_time  = ts.end_time()
    # 예측 구간에 해당하는 인덱스 생성
    new_times  = pd.date_range(start=last_time + ts.freq, periods=n, freq=freq)
    # 컬럼 구조 그대로 0 으로 채움
    df_pad     = pd.DataFrame(0.0, index=new_times, columns=ts.columns)
    ts_pad     = TimeSeries.from_dataframe(df_pad, fill_missing_dates=False, freq=freq)
    # 기존 + 패딩 합치기
    return ts.append(ts_pad)

future_cov_extended = [extend_covariate(ts, test[0].n_timesteps) for ts in future_cov]
past_cov = None if not past_cov else past_cov


def predict_with_prob_feedback_chunks(
    model,
    series_list: List[TimeSeries],
    past_covariates_list: Optional[List[TimeSeries]],
    future_covariates_list: Optional[List[TimeSeries]],
    n_steps: int,
    chunk_size: int = 90,         # = output_chunk_length
) -> List[TimeSeries]:
    """
    AR 루프로 n_steps 예측 → is_event는 sigmoid 처리 → history에 append.
    마지막 청크는 항상 chunk_size 예측 후 필요한 부분만 사용.
    """
    # 1) 초기화
    histories    = [s.copy() for s in series_list]
    preds_buffer = [[] for _ in series_list]
    icl = model.input_chunk_length  # input_chunk_length

    # 전체 '완전 청크' 횟수 + 마지막 잔여 계산
    num_full_chunks = n_steps // chunk_size + 1
    remainder = n_steps % chunk_size

    # 예측 결과를 버퍼에 쌓고 histories를 갱신하는 헬퍼
    def _append_preds_and_update_history(histories, batch_preds):
        new_histories = []
        for i, ts_pred in enumerate(batch_preds):
            df_pred = ts_pred.to_dataframe()
            probs   = torch.sigmoid(torch.tensor(df_pred["is_event"].values)).numpy()

            # 버퍼에 [demand, discount_pct, prob] 저장
            for j in range(len(df_pred)):
                preds_buffer[i].append([
                    float(df_pred["demand"].iat[j]),
                    float(df_pred["discount_pct"].iat[j]),
                    float(probs[j])
                ])

            # TimeSeries로 변환 후 history에 append
            df_new = pd.DataFrame({
                "demand":       df_pred["demand"].values,
                "discount_pct": df_pred["discount_pct"].values,
                "is_event":     probs
            }, index=ts_pred.time_index)

            new_ts = TimeSeries.from_dataframe(
                df_new,
                fill_missing_dates=False,
                freq=histories[i].time_index.freqstr
            )
            new_histories.append(histories[i].append(new_ts))
        return new_histories

    # 2) 완전 chunk_size 예측 반복
    for i in tqdm(range(num_full_chunks), total=num_full_chunks, desc="Full chunks"):
        clear_output(wait=True)

        batch_preds = model.predict(
            n=chunk_size,
            series=histories,
            past_covariates=past_covariates_list,
            future_covariates=future_covariates_list,
            verbose=False,
            random_state=42,
        )

        if i == num_full_chunks - 1:
            batch_preds = [ts_pred[:remainder] for ts_pred in batch_preds]

        histories = _append_preds_and_update_history(histories, batch_preds)
        
        print(i + 1,"/",num_full_chunks)

    # 4) preds_buffer → TimeSeries 결과 리스트 생성
    result = []
    for i, buf in enumerate(preds_buffer):
        start = series_list[i].end_time() + series_list[i].freq
        times = pd.date_range(start=start, periods=n_steps, freq=series_list[i].freq)
        df_out = pd.DataFrame(buf, index=times, columns=series_list[i].components)
        result.append(TimeSeries.from_dataframe(df_out))
    return result

best_tide = TiDEModel.load_from_checkpoint(model_name='TiDE_model', best=True)
n         = test[0].n_timesteps     # 731
preds_tide = predict_with_prob_feedback_chunks(
    model                  = best_tide,
    series_list            = train_val,
    past_covariates_list   = past_cov,
    future_covariates_list = future_cov_extended,
    n_steps                = n,
    chunk_size             = best_tide.output_chunk_length  # 90
)

from tqdm import tqdm

group_cols = ["sku","city"]  # 예시
groups_df = (
    dataset
    .loc[:, group_cols]
    .drop_duplicates()
    .sort_values(by=group_cols)   # from_group_dataframe 도 내부적으로 정렬하므로
    .reset_index(drop=True)
)

result = []
for i in tqdm(range(len(groups_df))):
    group_id = groups_df.iloc[i]
    
    pred = preds_tide[i].to_dataframe()
    pred = pred.reset_index()
    pred = pred.rename(columns={"index": "date"})
    # pred = pred.drop(columns=['discount_pct'])
    
    for j in range(len(pred)):
        result.append({
            "sku":  group_id["sku"],
            "city": group_id["city"],
            "date": pred['date'][j],
            "mean": pred['demand'][j],
            "discount_pct": pred['discount_pct'][j],
            "is_event": pred['is_event'][j],
        })

result_df = pd.DataFrame(result)
result_df['mean'] = np.expm1(result_df['mean'])
result_df['mean'] = result_df['mean'].round().astype(int)
result_df['date'] = pd.to_datetime(result_df['date'])
sub = pd.read_csv("extracted_contents/data/forecast_submission_template.csv", parse_dates=["date"])
sub.drop(columns=['mean'], inplace=True)
sub = sub.merge(result_df, on=['sku', 'city', 'date'], how='left')
sub.to_csv("result.csv", index=False)
sub.drop(columns=['discount_pct', 'is_event'], inplace=True)
sub.to_csv("forecast_submission_template.csv", index=False)