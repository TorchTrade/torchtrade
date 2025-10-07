# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import functools
import tempfile
from contextlib import nullcontext

import torch
from tensordict.nn import InteractionType, TensorDictModule

from torch import nn, optim
from torchrl.collectors import SyncDataCollector
from torchrl.data import (
    Composite,
    TensorDictPrioritizedReplayBuffer,
    TensorDictReplayBuffer,
)
from torch.distributions import Categorical

from torchrl.data.replay_buffers.storages import LazyMemmapStorage
from torchrl.envs import (
    Compose,
    DoubleToFloat,
    EnvCreator,
    InitTracker,
    ParallelEnv,
    RewardSum,
    TransformedEnv,
)
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.modules import MLP, SafeModule, SafeSequential

from torchrl.modules.distributions import OneHotCategorical

from torchrl.modules.tensordict_module.actors import ProbabilisticActor
from torchrl.objectives import SoftUpdate
from torchrl.objectives.sac import DiscreteSACLoss
from torchtrade.envs.offline.seqlongonly import SeqLongOnlyEnv, SeqLongOnlyEnvConfig
from torchtrade.envs.offline.utils import TimeFrame, TimeFrameUnit
from torchrl.trainers.helpers.models import ACTIVATIONS
from trading_nets.architectures.tabl.tabl import BiNMTABLModel
import copy
import ta
import numpy as np
import pandas as pd

# ====================================================================
# Environment utils
# -----------------



def custom_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess OHLCV dataframe with engineered features for RL trading.

    Expected columns: ["open", "high", "low", "close", "volume"]
    Index can be datetime or integer.
    """

    df = df.copy().reset_index(drop=False)

    # --- Basic features ---
    # Log returns
    df["features_return_log"] = np.log(df["close"]).diff()

    # Rolling volatility (5-period)
    df["features_volatility"] = df["features_return_log"].rolling(window=5).std()

    # ATR (14) normalized
    df["features_atr"] = ta.volatility.AverageTrueRange(
        high=df["high"], low=df["low"], close=df["close"], window=14
    ).average_true_range() / df["close"]

    # --- Momentum & trend ---
    ema_12 = ta.trend.EMAIndicator(close=df["close"], window=12).ema_indicator()
    ema_24 = ta.trend.EMAIndicator(close=df["close"], window=24).ema_indicator()
    df["features_ema_12"] = ema_12
    df["features_ema_24"] = ema_24
    df["features_ema_slope"] = ema_12.diff()

    macd = ta.trend.MACD(close=df["close"], window_slow=26, window_fast=12, window_sign=9)
    df["features_macd_hist"] = macd.macd_diff()

    df["features_rsi_14"] = ta.momentum.RSIIndicator(close=df["close"], window=14).rsi()

    # --- Volatility bands ---
    bb = ta.volatility.BollingerBands(close=df["close"], window=20, window_dev=2)
    df["features_bb_pct"] = bb.bollinger_pband()

    # --- Volume / flow ---
    df["features_volume_z"] = (
        (df["volume"] - df["volume"].rolling(20).mean()) /
        df["volume"].rolling(20).std()
    )
    df["features_vwap_dev"] = df["close"] - (
        (df["close"] * df["volume"]).cumsum() / df["volume"].cumsum()
    )

    # --- Candle structure ---
    df["features_body_ratio"] = (df["close"] - df["open"]) / (df["high"] - df["low"] + 1e-9)
    df["features_upper_wick"] = (df["high"] - df[["close", "open"]].max(axis=1)) / (
        df["high"] - df["low"] + 1e-9
    )
    df["features_lower_wick"] = (df[["close", "open"]].min(axis=1) - df["low"]) / (
        df["high"] - df["low"] + 1e-9
    )

    # Drop rows with NaN from indicators
    #df.dropna(inplace=True)
    df.fillna(0, inplace=True)


    return df


def env_maker(df, cfg, device="cpu"):

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
    )
    return SeqLongOnlyEnv(df, config, feature_preprocessing_fn=custom_preprocessing)



def apply_env_transforms(
    env,
):
    transformed_env = TransformedEnv(
        env,
        Compose(
            InitTracker(),
            DoubleToFloat(),
            RewardSum(),
        ),
    )
    return transformed_env


def make_environment(train_df, test_df, cfg, train_num_envs=1, eval_num_envs=1):
    """Make environments for training and evaluation."""
    maker = functools.partial(env_maker, train_df, cfg)
    parallel_env = ParallelEnv(
        train_num_envs,
        EnvCreator(maker),
        serial_for_single=True,
    )
    parallel_env.set_seed(cfg.env.seed)

    train_env = apply_env_transforms(parallel_env)

    maker = functools.partial(env_maker, test_df, cfg)
    eval_env = TransformedEnv(
        ParallelEnv(
            eval_num_envs,
            EnvCreator(maker),
            serial_for_single=True,
        ),
        train_env.transform.clone(),
    )
    return train_env, eval_env



# ====================================================================
# Collector and replay buffer
# ---------------------------


def make_collector(cfg, train_env, actor_model_explore, compile_mode):
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
        init_random_frames=cfg.collector.init_random_frames,
        max_frames_per_traj=cfg.collector.max_frames_per_traj,
        total_frames=cfg.collector.total_frames,
        device=device,
        compile_policy={"mode": compile_mode} if compile_mode else False,
        cudagraph_policy={"warmup": 10} if cfg.compile.cudagraphs else False,
    )
    collector.set_seed(cfg.env.seed)
    return collector


def make_replay_buffer(
    batch_size,
    prb=False,
    buffer_size=1000000,
    scratch_dir=None,
    device="cpu",
    prefetch=3,
):
    with (
        tempfile.TemporaryDirectory()
        if scratch_dir is None
        else nullcontext(scratch_dir)
    ) as scratch_dir:
        if prb:
            replay_buffer = TensorDictPrioritizedReplayBuffer(
                alpha=0.7,
                beta=0.5,
                pin_memory=False,
                prefetch=prefetch,
                storage=LazyMemmapStorage(
                    buffer_size,
                    scratch_dir=scratch_dir,
                    device=device,
                ),
                batch_size=batch_size,
            )
        else:
            replay_buffer = TensorDictReplayBuffer(
                pin_memory=False,
                prefetch=prefetch,
                storage=LazyMemmapStorage(
                    buffer_size,
                    scratch_dir=scratch_dir,
                    device=device,
                ),
                batch_size=batch_size,
            )
        return replay_buffer


# ====================================================================
# Model
# -----

def make_sac_agent(cfg, env, device):
    """Make discrete IQL agent."""
    # Define Actor Network
    action_spec = env.action_spec
    market_data_keys = [k for k in list(env.observation_spec.keys()) if k.startswith("market_data")]
    assert "account_state" in list(env.observation_spec.keys()), "Account state key not in observation spec"
    account_state_key = "account_state"
    # Define Actor Network
    time_frames = cfg.env.time_frames
    window_sizes = cfg.env.window_sizes
    freqs = cfg.env.freqs
    assert len(time_frames) == len(market_data_keys), f"Amount of time frames {len(time_frames)} and env market data keys do not match! Keys: {market_data_keys}"
    encoders = []
    
    # Build the encoder
    for key, t, w, fre in zip(market_data_keys, time_frames, window_sizes, freqs):
    
        model = BiNMTABLModel(input_shape=(1, w, 14),
                            output_shape=(1, 14), # if None, the output shape will be the same as the input shape otherwise you have to provide the output shape (out_seq, out_feat)
                            hidden_seq_size=w,
                            hidden_feature_size=14,
                            num_heads=3,
                            activation="relu",
                            final_activation="relu",
                            dropout=0.1,
                            initializer="kaiming_uniform")
        encoders.append(SafeModule(
            module=model,
            in_keys=key,
            out_keys=[f"encoding_{t}_{fre}_{w}"],
        ).to(device))

    account_state_encoder = SafeModule(
        module=MLP(
            num_cells=[32],
            out_features=14,
            activation_class=ACTIVATIONS[cfg.network.activation],
            device=device,
        ),
        in_keys=[account_state_key],
        out_keys=["encoding_account_state"],
    )

    encoder = SafeSequential(*encoders, account_state_encoder)

    # Define the actor
    actor_net = MLP(
        num_cells=cfg.network.hidden_sizes,
        out_features=3,
        activation_class=ACTIVATIONS[cfg.network.activation],
        device=device,
    )

    actor_module = SafeModule(
        module=actor_net,
        in_keys=[f"encoding_{t}_{fre}_{w}" for t, w, fre in zip(time_frames, window_sizes, freqs)] + ["encoding_account_state"],
        out_keys=["logits"],
    )
    full_actor = SafeSequential(encoder, actor_module)
    
    actor = ProbabilisticActor(
        spec=Composite(action=action_spec).to(device),
        module=full_actor,
        in_keys=["logits"],
        out_keys=["action"],
        distribution_class=Categorical, #Categorical, OneHotCategorical
        distribution_kwargs={},
        default_interaction_type=InteractionType.RANDOM,
        return_log_prob=False,
    )

    # Define Critic Network
    qvalue_net = MLP(
        num_cells=cfg.network.hidden_sizes,
        out_features=3,
        activation_class=ACTIVATIONS[cfg.network.activation],
        device=device,
    )
    
    qvalue = SafeModule(
        module=qvalue_net,
        in_keys=[f"encoding_{t}_{fre}_{w}" for t, w, fre in zip(time_frames, window_sizes, freqs)] + ["encoding_account_state"],
        out_keys=["action_value"],
    )
    full_qvalue = SafeSequential(copy.deepcopy(encoder), qvalue)

    # Define complete model
    model = torch.nn.ModuleList([actor, full_qvalue])

    # init nets
    example_td = env.fake_tensordict().to(device)
    with torch.no_grad(), set_exploration_type(ExplorationType.RANDOM):
        td = example_td
        for net in model:
            net(td)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {total_params}")

    return model

# ====================================================================
# Discrete SAC Loss
# ---------


def make_loss_module(cfg, model):
    """Make loss module and target network updater."""
    # Create discrete SAC loss
    loss_module = DiscreteSACLoss(
        actor_network=model[0],
        qvalue_network=model[1],
        num_actions=model[0].spec["action"].space.n,
        num_qvalue_nets=2,
        loss_function=cfg.optim.loss_function,
        target_entropy_weight=cfg.optim.target_entropy_weight,
        delay_qvalue=True,
    )
    loss_module.make_value_estimator(gamma=cfg.optim.gamma)

    # Define Target Network Updater
    target_net_updater = SoftUpdate(loss_module, eps=cfg.optim.target_update_polyak)
    return loss_module, target_net_updater


def make_optimizer(cfg, loss_module):
    critic_params = list(loss_module.qvalue_network_params.flatten_keys().values())
    actor_params = list(loss_module.actor_network_params.flatten_keys().values())

    optimizer_actor = optim.Adam(
        actor_params,
        lr=cfg.optim.lr,
        weight_decay=cfg.optim.weight_decay,
    )
    optimizer_critic = optim.Adam(
        critic_params,
        lr=cfg.optim.lr,
        weight_decay=cfg.optim.weight_decay,
    )
    optimizer_alpha = optim.Adam(
        [loss_module.log_alpha],
        lr=3.0e-4,
    )
    return optimizer_actor, optimizer_critic, optimizer_alpha


# ====================================================================
# General utils
# ---------


def log_metrics(logger, metrics, step):
    for metric_name, metric_value in metrics.items():
        logger.log_scalar(metric_name, metric_value, step)

