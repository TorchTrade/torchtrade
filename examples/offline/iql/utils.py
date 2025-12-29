from __future__ import annotations

import functools

import torch.nn
import torch.optim
import tensordict
from tensordict.nn import InteractionType
from torch.distributions import Categorical
from torchrl.data import Categorical as CategoricalSpec

from torchrl.collectors import SyncDataCollector
from torchrl.data import (
    Composite,
    LazyMemmapStorage,
    TensorDictPrioritizedReplayBuffer,
    TensorDictReplayBuffer,
)
from torchrl.data.replay_buffers import SamplerWithoutReplacement
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
from torchrl.modules import (
    MLP,
    ProbabilisticActor,
    SafeModule,
    SafeSequential,
)
from torchrl.objectives import DiscreteIQLLoss, HardUpdate
from torchrl.record import VideoRecorder
from torchrl.trainers.helpers.models import ACTIVATIONS
from trading_nets.architectures.tabl.tabl import BiNMTABLModel
from trading_nets.architectures.wavenet.simple_1d_wave import Simple1DWaveEncoder

import copy
import pandas as pd
import numpy as np
import ta
from torchtrade.envs.offline.seqlongonly import SeqLongOnlyEnv, SeqLongOnlyEnvConfig
from torchtrade.envs.offline.utils import TimeFrame, TimeFrameUnit
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


def make_offline_replay_buffer(rb_cfg):
    if rb_cfg.data_path == "synthetic":
        # Generate synthetic data for testing
        import torch
        from tensordict import TensorDict
        n_transitions = rb_cfg.buffer_size
        obs_dim = 4
        window_size = 12
        n_actions = 3

        td = TensorDict({
            "observation": torch.randn(n_transitions, window_size, obs_dim),
            "action": torch.randint(0, n_actions, (n_transitions,)),
            "next": TensorDict({
                "observation": torch.randn(n_transitions, window_size, obs_dim),
                "reward": torch.randn(n_transitions) * 0.01,
                "done": torch.zeros(n_transitions, dtype=torch.bool),
                "terminated": torch.zeros(n_transitions, dtype=torch.bool),
            }, batch_size=[n_transitions]),
        }, batch_size=[n_transitions])
    elif "/" in rb_cfg.data_path and not rb_cfg.data_path.startswith("/"):
        # HuggingFace dataset path (e.g., "Torch-Trade/AlpacaLiveData_LongOnly-v0")
        from datasets import load_dataset
        from torchtrade.utils import dataset_to_td
        ds = load_dataset(rb_cfg.data_path, split="train")
        td = dataset_to_td(ds)
    else:
        td = tensordict.load(rb_cfg.data_path)

    size = td.shape[0]
    data = TensorDictReplayBuffer(
        pin_memory=False,
        prefetch=4,
        #split_trajs=False,
        storage=LazyMemmapStorage(size),
        batch_size=rb_cfg.batch_size,
        sampler=SamplerWithoutReplacement(drop_last=True),
    )
    data.extend(td)
    del td

    # add reward2go if needed


    data.append_transform(DoubleToFloat())

    return data


# ====================================================================
# Model
# -----
#
# We give one version of the model for learning from pixels, and one for state.
# TorchRL comes in handy at this point, as the high-level interactions with
# these models is unchanged, regardless of the modality.
#

def make_discrete_iql_wavenet_model(cfg, env, device):
    """Make discrete IQL agent."""
    # Define Actor Network
    market_data_keys = [k for k in list(env.observation_spec.keys()) if k.startswith("market_data")]
    assert "account_state" in list(env.observation_spec.keys()), "Account state key not in observation spec"
    account_state_key = "account_state"
    # Define Actor Network
    time_frames = cfg.env.time_frames
    assert len(time_frames) == len(market_data_keys), f"Amount of time frames {len(time_frames)} and env market data keys do not match! Keys: {market_data_keys}"
    encoders = []
    
    # Build the encoder
    for key, freq, t in zip(market_data_keys, cfg.env.freqs, cfg.env.time_frames):
        net = Simple1DWaveEncoder(feature_dim=14,
                                base_channels=32,
                                num_layers=4,
                                out_channels=14,
                                squeeze_output=True,
                                dil_norm_type='layernorm'
                                )
        encoders.append(SafeModule(net, in_keys=key, out_keys=[f"encoding{t}{freq.lower()}"]))

    account_state_encoder = SafeModule(
        module=MLP(
            num_cells=[32],
            out_features=14,
            activation_class=ACTIVATIONS[cfg.model.activation],
            device=device,
        ),
        in_keys=["account_state"],
        out_keys=["encoding_account_state"],
    )


    encoder = SafeSequential(*encoders, account_state_encoder).to(device)
    
    actor_net = MLP(
        num_cells=cfg.model.hidden_sizes,
        out_features=3,
        activation_class=ACTIVATIONS[cfg.model.activation],
        device=device,
    )

    actor_module = SafeModule(
        module=actor_net,
        in_keys=["encoding1min", "encoding5min", "encoding15min", "encoding1h", "encoding_account_state"],
        out_keys=["logits"],
    )
    full_actor = SafeSequential(encoder, actor_module)
    
    actor = ProbabilisticActor(
        spec=Composite(action=env.full_action_spec_unbatched).to(device),
        module=full_actor,
        in_keys=["logits"],
        out_keys=["action"],
        distribution_class=Categorical,
        distribution_kwargs={},
        default_interaction_type=InteractionType.RANDOM,
        return_log_prob=False,
    )

    # Define Critic Network
    qvalue_net = MLP(
        num_cells=cfg.model.hidden_sizes,
        out_features=3,
        activation_class=ACTIVATIONS[cfg.model.activation],
        device=device,
    )
    
    qvalue = SafeModule(
        module=qvalue_net,
        in_keys=["encoding1min", "encoding5min", "encoding15min", "encoding1h", "encoding_account_state"],
        out_keys=["state_action_value"],
    )
    full_qvalue = SafeSequential(copy.deepcopy(encoder), qvalue)

    # Define Value Network
    value_net = MLP(
        num_cells=cfg.model.hidden_sizes,
        out_features=1,
        activation_class=ACTIVATIONS[cfg.model.activation],
        device=device,
    )
    value_net = SafeModule(
        module=value_net,
        in_keys=["encoding1min", "encoding5min", "encoding15min", "encoding1h", "encoding_account_state"],
        out_keys=["state_value"],
    )   
    full_value = SafeSequential(copy.deepcopy(encoder), value_net)

    model = torch.nn.ModuleList([actor, full_qvalue, full_value])


    # init nets

    example_td = tensordict.TensorDict(
        {
            "market_data_1Minute_12": torch.randn(1, 12, 14),
            "market_data_5Minute_8": torch.randn(1, 8, 14),
            "market_data_15Minute_8": torch.randn(1, 8, 14),
            "market_data_1Hour_24": torch.randn(1, 24, 14),
            "account_state": torch.randn(1, 6),
        }
    ).to(device)
    with torch.no_grad(), set_exploration_type(ExplorationType.RANDOM):
        td = example_td
        for net in model:
            net(td)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {total_params}")

    return model



def make_discrete_iql_binmtabl_model(cfg, device):
    """Make discrete IQL agent."""
    # Define Actor Network
    action_spec = CategoricalSpec(3)
    # Define Actor Network
    import tensordict
    encodernet1min12 = BiNMTABLModel(input_shape=(12, 14),
                        output_shape=(1, 14), # if None, the output shape will be the same as the input shape otherwise you have to provide the output shape (out_seq, out_feat)
                        hidden_seq_size=12,
                        hidden_feature_size=14,
                        num_heads=3,
                        activation="relu",
                        final_activation="relu",
                        dropout=0.1,
                        initializer="kaiming_uniform")
    encodernet5min8 = BiNMTABLModel(input_shape=(8, 14),
                        output_shape=(1, 14), # if None, the output shape will be the same as the input shape otherwise you have to provide the output shape (out_seq, out_feat)
                        hidden_seq_size=8,
                        hidden_feature_size=14,
                        num_heads=3,
                        activation="relu",
                        final_activation="relu",
                        dropout=0.1,
                        initializer="kaiming_uniform")

    encodernet15min8 = BiNMTABLModel(input_shape=(8, 14),
                        output_shape=(1, 14), # if None, the output shape will be the same as the input shape otherwise you have to provide the output shape (out_seq, out_feat)
                        hidden_seq_size=8,
                        hidden_feature_size=14,
                        num_heads=3,
                        activation="relu",
                        final_activation="relu",
                        dropout=0.1,
                        initializer="kaiming_uniform")

    encodernet1h24 = BiNMTABLModel(input_shape=(24, 14),
                        output_shape=(1, 14), # if None, the output shape will be the same as the input shape otherwise you have to provide the output shape (out_seq, out_feat)
                        hidden_seq_size=24,
                        hidden_feature_size=14,
                        num_heads=3,
                        activation="relu",
                        final_activation="relu",
                        dropout=0.1,
                        initializer="kaiming_uniform")

    encoder1min12 = SafeModule(
        module=encodernet1min12,
        in_keys=["market_data_1Minute_12"],
        out_keys=["encoding1min"],
    ).to(device)
    encoder5min8 = SafeModule(
        module=encodernet5min8,
        in_keys=["market_data_5Minute_8"],
        out_keys=["encoding5min"],
    ).to(device)
    encoder15min8 = SafeModule(
        module=encodernet15min8,
        in_keys=["market_data_15Minute_8"],
        out_keys=["encoding15min"],
    ).to(device)
    encoder1h24 = SafeModule(
        module=encodernet1h24,
        in_keys=["market_data_1Hour_24"],
        out_keys=["encoding1h"],

    ).to(device)
    account_state_encoder = SafeModule(
        module=MLP(
            num_cells=[32],
            out_features=14,
            activation_class=ACTIVATIONS[cfg.model.activation],
            device=device,
        ),
        in_keys=["account_state"],
        out_keys=["encoding_account_state"],
    )

    encoder = SafeSequential(encoder1min12, encoder5min8, encoder15min8, encoder1h24, account_state_encoder)

    actor_net = MLP(
        num_cells=cfg.model.hidden_sizes,
        out_features=3,
        activation_class=ACTIVATIONS[cfg.model.activation],
        device=device,
    )

    actor_module = SafeModule(
        module=actor_net,
        in_keys=["encoding1min", "encoding5min", "encoding15min", "encoding1h", "encoding_account_state"],
        out_keys=["logits"],
    )
    full_actor = SafeSequential(encoder, actor_module)
    
    actor = ProbabilisticActor(
        spec=Composite(action=action_spec).to(device),
        module=full_actor,
        in_keys=["logits"],
        out_keys=["action"],
        distribution_class=Categorical,
        distribution_kwargs={},
        default_interaction_type=InteractionType.RANDOM,
        return_log_prob=False,
    )

    # Define Critic Network
    qvalue_net = MLP(
        num_cells=cfg.model.hidden_sizes,
        out_features=3,
        activation_class=ACTIVATIONS[cfg.model.activation],
        device=device,
    )
    
    qvalue = SafeModule(
        module=qvalue_net,
        in_keys=["encoding1min", "encoding5min", "encoding15min", "encoding1h", "encoding_account_state"],
        out_keys=["state_action_value"],
    )
    full_qvalue = SafeSequential(copy.deepcopy(encoder), qvalue)

    # Define Value Network
    value_net = MLP(
        num_cells=cfg.model.hidden_sizes,
        out_features=1,
        activation_class=ACTIVATIONS[cfg.model.activation],
        device=device,
    )
    value_net = SafeModule(
        module=value_net,
        in_keys=["encoding1min", "encoding5min", "encoding15min", "encoding1h", "encoding_account_state"],
        out_keys=["state_value"],
    )   
    full_value = SafeSequential(copy.deepcopy(encoder), value_net)

    model = torch.nn.ModuleList([actor, full_qvalue, full_value])


    # init nets

    example_td = tensordict.TensorDict(
        {
            "market_data_1Minute_12": torch.randn(1, 12, 14),
            "market_data_5Minute_8": torch.randn(1, 8, 14),
            "market_data_15Minute_8": torch.randn(1, 8, 14),
            "market_data_1Hour_24": torch.randn(1, 24, 14),
            "account_state": torch.randn(1, 6),
        }
    ).to(device)
    with torch.no_grad(), set_exploration_type(ExplorationType.RANDOM):
        td = example_td
        for net in model:
            net(td)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {total_params}")

    return model


# ====================================================================
# IQL Loss
# ---------

def make_discrete_loss(loss_cfg, model, device):
    loss_module = DiscreteIQLLoss(
        model[0],
        model[1],
        value_network=model[2],
        loss_function=loss_cfg.loss_function,
        temperature=loss_cfg.temperature,
        expectile=loss_cfg.expectile,
        action_space="categorical",
    )
    loss_module.make_value_estimator(gamma=loss_cfg.gamma, device=device)
    target_net_updater = HardUpdate(
        loss_module, value_network_update_interval=loss_cfg.hard_update_interval
    )

    return loss_module, target_net_updater


def make_iql_optimizer(optim_cfg, loss_module):
    critic_params = list(loss_module.qvalue_network_params.flatten_keys().values())
    actor_params = list(loss_module.actor_network_params.flatten_keys().values())
    value_params = list(loss_module.value_network_params.flatten_keys().values())

    optimizer_actor = torch.optim.Adam(
        actor_params,
        lr=optim_cfg.lr,
        weight_decay=optim_cfg.weight_decay,
    )
    optimizer_critic = torch.optim.Adam(
        critic_params,
        lr=optim_cfg.lr,
        weight_decay=optim_cfg.weight_decay,
    )
    optimizer_value = torch.optim.Adam(
        value_params,
        lr=optim_cfg.lr,
        weight_decay=optim_cfg.weight_decay,
    )
    return optimizer_actor, optimizer_critic, optimizer_value


# ====================================================================
# General utils
# ---------


def log_metrics(logger, metrics, step):
    if logger is not None:
        for metric_name, metric_value in metrics.items():
            logger.log_scalar(metric_name, metric_value, step)


def dump_video(module):
    if isinstance(module, VideoRecorder):
        module.dump()