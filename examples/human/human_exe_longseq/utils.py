from __future__ import annotations
import functools
import torch
from torchrl.envs import (
    DoubleToFloat,
    EnvCreator,
    ParallelEnv,
    RewardSum,
    InitTracker,
    Compose,
    TransformedEnv,
    StepCounter,
)
from torchrl.collectors import SyncDataCollector

from torchtrade.envs.offline.seqlongonly import SeqLongOnlyEnv, SeqLongOnlyEnvConfig
from torchtrade.envs.offline.utils import TimeFrame, TimeFrameUnit
import pandas as pd

# ====================================================================
# Environment utils
# --------------------------------------------------------------------


def custom_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess OHLCV dataframe with engineered features for RL trading.

    Expected columns: ["open", "high", "low", "close", "volume"]
    Index can be datetime or integer.
    """

    df = df.copy().reset_index(drop=False)

    # --- Basic features ---


    # --- Momentum & trend ---
    ema_12 = ta.trend.EMAIndicator(close=df["close"], window=12).ema_indicator()
    ema_24 = ta.trend.EMAIndicator(close=df["close"], window=24).ema_indicator()
    df["features_ema_12"] = ema_12
    df["features_ema_24"] = ema_24

    df["features_close"] = df["close"]

    df.fillna(0, inplace=True)


    return df


def env_maker(df, cfg, device="cpu", max_traj_length=1, random_start=False):

    # TODO: Make this configurable with config
    time_frames=[
        TimeFrame(1, TimeFrameUnit.Minute),
        TimeFrame(5, TimeFrameUnit.Minute),
        TimeFrame(15, TimeFrameUnit.Minute),
        TimeFrame(1, TimeFrameUnit.Hour),
    ]
    window_sizes=[12, 8, 8, 24]  # ~12m, 40m, 2h, 1d
    execute_on=TimeFrame(5, TimeFrameUnit.Minute) # Try 15min

    config = SeqLongOnlyEnvConfig(
        symbol=cfg.env.symbol,
        time_frames=time_frames,
        window_sizes=window_sizes,
        execute_on=execute_on,
        include_base_features=False,
        initial_cash=cfg.env.initial_cash,
        slippage=cfg.env.slippage,
        transaction_fee=cfg.env.transaction_fee,
        bankrupt_threshold=cfg.env.bankrupt_threshold,
        seed=cfg.env.seed,
        max_traj_length=max_traj_length,
        random_start=random_start
    )
    return SeqLongOnlyEnv(df, config, feature_preprocessing_fn=custom_preprocessing)



def apply_env_transforms(
    env,
    max_steps,
):
    transformed_env = TransformedEnv(
        env,
        Compose(
            InitTracker(),
            DoubleToFloat(),
            RewardSum(),
            StepCounter(max_steps=max_steps),
        ),
    )
    return transformed_env


def make_environment(train_df, cfg, train_num_envs=1, 
                     max_train_traj_length=1):
    """Make environments for training and evaluation."""
    maker = functools.partial(env_maker, train_df, cfg, max_traj_length=max_train_traj_length, random_start=True)
    max_train_steps = train_df.shape[0]
    parallel_env = ParallelEnv(
        train_num_envs,
        EnvCreator(maker),
        serial_for_single=True,
    )
    parallel_env.set_seed(cfg.env.seed)

    train_env = apply_env_transforms(parallel_env, max_train_steps)

    return train_env


def make_collector(cfg, train_env, actor_model_explore):
    """Make collector."""
    device = cfg.collector.device
    if device in ("", None):
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
        else:
            device = torch.device("cpu")
    collector = SyncDataCollector(
        train_env,
        actor_model_explore,
        frames_per_batch=cfg.collector.frames_per_batch,
        total_frames=cfg.collector.total_frames,
        device=device,
    )
    collector.set_seed(cfg.env.seed)
    return collector



def log_metrics(logger, metrics, step):
    for metric_name, metric_value in metrics.items():
        logger.log_scalar(metric_name, metric_value, step)