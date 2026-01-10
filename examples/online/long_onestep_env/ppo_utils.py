# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations
import functools

from pandas.tseries import frequencies
import torch.nn
from tensordict.nn import TensorDictModule
from torchrl.envs import (
    DoubleToFloat,
    EnvCreator,
    ExplorationType,
    ParallelEnv,
    RewardSum,
    InitTracker,
    Compose,
    TransformedEnv,
    StepCounter,
    VecNormV2,
)
from torchrl.collectors import SyncDataCollector

from torchrl.modules import (
    ActorValueOperator,
    MLP,
    ProbabilisticActor,
    ValueOperator,
    SafeModule,
    SafeSequential,
)
from torchtrade.models.simple_encoders import SimpleCNNEncoder, SimpleMLPEncoder

from torchtrade.envs import LongOnlyOneStepEnv, LongOnlyOneStepEnvConfig, SeqLongOnlySLTPEnvConfig, SeqLongOnlySLTPEnv
from torchtrade.envs.offline.utils import TimeFrame, TimeFrameUnit, get_timeframe_unit
import ta
import numpy as np
import pandas as pd
from torchrl.trainers.helpers.models import ACTIVATIONS

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
    # Log returns
    # df["features_return_log"] = np.log(df["close"]).diff()
    df["features_close"] = df["close"]
    df["features_open"] = df["open"]
    df["features_high"] = df["high"]
    df["features_low"] = df["low"]
    df["features_volume"] = df["volume"]

    # # Rolling volatility (5-period)
    # df["features_volatility"] = df["features_return_log"].rolling(window=5).std()

    # # ATR (14) normalized
    # df["features_atr"] = ta.volatility.AverageTrueRange(
    #     high=df["high"], low=df["low"], close=df["close"], window=14
    # ).average_true_range() / df["close"]

    # --- Momentum & trend ---
    # ema_12 = ta.trend.EMAIndicator(close=df["close"], window=12).ema_indicator()
    # ema_24 = ta.trend.EMAIndicator(close=df["close"], window=24).ema_indicator()
    # df["features_ema_12"] = ema_12
    # df["features_ema_24"] = ema_24
    #df["features_ema_slope"] = ema_12.diff()

    # macd = ta.trend.MACD(close=df["close"], window_slow=26, window_fast=12, window_sign=9)
    # df["features_macd_hist"] = macd.macd_diff()

    # df["features_rsi_14"] = ta.momentum.RSIIndicator(close=df["close"], window=14).rsi()

    # # --- Volatility bands ---
    # bb = ta.volatility.BollingerBands(close=df["close"], window=20, window_dev=2)
    # df["features_bb_pct"] = bb.bollinger_pband()

    # --- Volume / flow ---
    # df["features_volume_z"] = (
    #     (df["volume"] - df["volume"].rolling(20).mean()) /
    #     df["volume"].rolling(20).std()
    # )
    # df["features_vwap_dev"] = df["close"] - (
    #     (df["close"] * df["volume"]).cumsum() / df["volume"].cumsum()
    # )

    # # --- Candle structure ---
    # df["features_body_ratio"] = (df["close"] - df["open"]) / (df["high"] - df["low"] + 1e-9)
    # df["features_upper_wick"] = (df["high"] - df[["close", "open"]].max(axis=1)) / (
    #     df["high"] - df["low"] + 1e-9
    # )
    # df["features_lower_wick"] = (df[["close", "open"]].min(axis=1) - df["low"]) / (
    #     df["high"] - df["low"] + 1e-9
    #)

    # Drop rows with NaN from indicators
    #df.dropna(inplace=True)
    df.fillna(0, inplace=True)


    return df

def env_maker(df, cfg, device="cpu", max_traj_length=1, eval=False):

    window_sizes = cfg.env.window_sizes
    execute_on = cfg.env.execute_on

    time_frames = [TimeFrame(t, get_timeframe_unit(f)) for t, f in zip(cfg.env.time_frames, cfg.env.freqs)]
    execute_on=TimeFrame(execute_on[0], get_timeframe_unit(execute_on[1]))

    if not eval:
        config = LongOnlyOneStepEnvConfig(
            symbol=cfg.env.symbol,
            time_frames=time_frames,
            window_sizes=window_sizes,
            execute_on=execute_on,
            include_base_features=False,
            initial_cash=cfg.env.initial_cash,
            slippage=cfg.env.slippage,
            transaction_fee=cfg.env.transaction_fee,
            bankrupt_threshold=cfg.env.bankrupt_threshold,
            stoploss_levels=cfg.env.stoploss_levels,
            takeprofit_levels=cfg.env.takeprofit_levels,
            seed=cfg.env.seed,
            max_traj_length=max_traj_length,
        )
        return LongOnlyOneStepEnv(df, config, feature_preprocessing_fn=custom_preprocessing)
    else:
        config = SeqLongOnlySLTPEnvConfig(
            symbol=cfg.env.symbol,
            time_frames=time_frames,
            window_sizes=window_sizes,
            execute_on=execute_on,
            include_base_features=False,
            initial_cash=cfg.env.initial_cash,
            slippage=cfg.env.slippage,
            transaction_fee=cfg.env.transaction_fee,
            bankrupt_threshold=cfg.env.bankrupt_threshold,
            stoploss_levels=cfg.env.stoploss_levels,
            takeprofit_levels=cfg.env.takeprofit_levels,
            seed=cfg.env.seed,
            max_traj_length=max_traj_length,
            random_start=False
        )
        return SeqLongOnlySLTPEnv(df, config, feature_preprocessing_fn=custom_preprocessing)

def apply_env_transforms(env, max_steps):
    """Apply standard transforms to the environment.

    Args:
        env: Base environment
        max_steps: Maximum steps for StepCounter

    Returns:
        transformed_env: Environment with transforms applied
        vecnorm: The VecNormV2 instance for potential statistics sharing
    """
    # Get observation keys for normalization (market_data_* and account_state)
    obs_keys = [k for k in env.observation_spec.keys() if k.startswith("market_data") or k == "account_state"]

    vecnorm = VecNormV2(
        in_keys=obs_keys,
        decay=0.99999,
        eps=1e-8,
    )

    transformed_env = TransformedEnv(
        env,
        Compose(
            InitTracker(),
            DoubleToFloat(),
            vecnorm,
            RewardSum(),
            StepCounter(max_steps=max_steps),
        ),
    )
    return transformed_env, vecnorm


def make_environment(train_df, test_df, cfg, train_num_envs=1, eval_num_envs=1,
                     max_train_traj_length=1,
                     max_eval_traj_length=1,
                     device="cpu"):
    """Make environments for training and evaluation."""
    maker = functools.partial(env_maker, train_df, cfg, max_traj_length=max_train_traj_length)
    max_train_steps = train_df.shape[0]
    parallel_env = ParallelEnv(
        train_num_envs,
        EnvCreator(maker),
        serial_for_single=True,
    )
    parallel_env.set_seed(cfg.env.seed)

    # Create train environment and get its VecNormV2 instance
    train_env, train_vecnorm = apply_env_transforms(parallel_env, max_train_steps)

    # Create eval environment with its own VecNormV2
    # Note: Each VecNormV2 will maintain its own running statistics
    # This is acceptable as eval will normalize observations consistently during evaluation
    maker = functools.partial(env_maker, test_df, cfg, max_traj_length=max_eval_traj_length, eval=True)
    eval_base_env = ParallelEnv(
        eval_num_envs,
        EnvCreator(maker),
        serial_for_single=True,
    )
    max_eval_steps = test_df.shape[0]

    # Create eval environment with separate VecNormV2
    eval_env, eval_vecnorm = apply_env_transforms(eval_base_env, max_eval_steps)

    # Move eval environment to device
    eval_env.to(device)

    # Explicitly move VecNormV2 internal statistics to device
    # This is needed because env.to(device) doesn't properly move transform statistics
    if hasattr(eval_vecnorm, '_loc') and eval_vecnorm._loc is not None:
        eval_vecnorm._loc = eval_vecnorm._loc.to(device)
    if hasattr(eval_vecnorm, '_var') and eval_vecnorm._var is not None:
        eval_vecnorm._var = eval_vecnorm._var.to(device)
    if hasattr(eval_vecnorm, '_count') and eval_vecnorm._count is not None:
        eval_vecnorm._count = eval_vecnorm._count.to(device)

    # Freeze eval VecNormV2 so it doesn't update statistics during evaluation
    eval_vecnorm.freeze()

    return train_env, eval_env, train_vecnorm, eval_vecnorm



# ====================================================================
# Model utils
# --------------------------------------------------------------------


def make_discrete_ppo_binmtabl_model(cfg, env, device):
    """Make discrete PPO agent."""
    # Define Actor Network
    activation = "tanh"
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

    num_features = env.observation_spec[market_data_keys[0]].shape[-1]
    
    # Build the encoder
    for key, t, w, fre in zip(market_data_keys, time_frames, window_sizes, freqs):
    
        model = SimpleCNNEncoder(input_shape=(w, num_features),
                            output_shape=(1, 14), # if None, the output shape will be the same as the input shape otherwise you have to provide the output shape (out_seq, out_feat)
                            hidden_channels=64,
                            kernel_size=3,
                            activation=activation,
                            final_activation=activation,
                            dropout=0.1)
        encoders.append(SafeModule(
            module=model,
            in_keys=key,
            out_keys=[f"encoding_{t}_{fre}_{w}"],
        ).to(device))


    account_encoder_model = SimpleMLPEncoder(
        input_shape=(1, 7),  # 7 account state features, single timestep
        output_shape=(1, 14),  # Match embedding_dim output
        hidden_sizes=(32, 32),
        activation="gelu",
        dropout=0.1,
        final_activation="gelu",
    )

    account_state_encoder = SafeModule(
        # module=MLP(
        #     num_cells=[32, 32],
        #     out_features=14,
        #     activation_class=ACTIVATIONS[activation],
        #     device=device,
        # ),
        module=account_encoder_model,
        in_keys=[account_state_key],
        out_keys=["encoding_account_state"],
    ).to(device)

    # Define the actor
    common = MLP(
        num_cells=[128, 128],
        out_features=128,
        activation_class=ACTIVATIONS[activation],
        device=device,
    )

    common_module = SafeModule(
        module=common,
        in_keys=[f"encoding_{t}_{fre}_{w}" for t, w, fre in zip(time_frames, window_sizes, freqs)] + ["encoding_account_state"],
        out_keys=["common_features"],
    )
    common_module = SafeSequential(*encoders, account_state_encoder, common_module)

    action_out_features = action_spec.n
    distribution_class = torch.distributions.Categorical
    distribution_kwargs = {}

    # Define on head for the policy
    policy_net = MLP(
        in_features=128,
        out_features=action_out_features,
        activation_class=ACTIVATIONS[activation],
        num_cells=[],
        device=device,
    )
    policy_module = SafeModule(
        module=policy_net,
        in_keys=["common_features"],
        out_keys=["logits"],
    )

    policy_module = ProbabilisticActor(
        policy_module,
        in_keys=["logits"],
        spec=env.full_action_spec_unbatched.to(device), 
        distribution_class=distribution_class,
        distribution_kwargs=distribution_kwargs,
        return_log_prob=True,
        default_interaction_type=ExplorationType.RANDOM,
    )

    # Define another head for the value
    value_net = MLP(
        activation_class=torch.nn.ReLU, #ACTIVATIONS[activation], #
        in_features=128,
        out_features=1,
        num_cells=[],
        device=device,
    )
    value_module = ValueOperator(
        value_net,
        in_keys=["common_features"],
    )

    return common_module, policy_module, value_module

def make_ppo_models(env, device, cfg):
    common_module, policy_module, value_module = make_discrete_ppo_binmtabl_model(
        cfg,
        env,
        device=device,
    )

    # Wrap modules in a single ActorCritic operator
    actor_critic = ActorValueOperator(
        common_operator=common_module,
        policy_operator=policy_module,
        value_operator=value_module,
    ).to(device)

    with torch.no_grad():
        td = env.fake_tensordict().unsqueeze(0).expand(3, 2).to(actor_critic.device)
        actor_critic(td)
        del td

    total_params = sum(p.numel() for p in actor_critic.parameters())
    print(f"Total number of parameters: {total_params}")

    actor = actor_critic.get_policy_operator()
    critic = actor_critic.get_value_operator()

    return actor, critic


# ====================================================================
# Evaluation utils
# --------------------------------------------------------------------



def eval_model(actor, test_env, num_episodes=3):
    test_rewards = []
    for _ in range(num_episodes):
        td_test = test_env.rollout(
            policy=actor,
            auto_reset=True,
            auto_cast_to_device=True,
            break_when_any_done=True,
            max_steps=10_000_000,
        )
        test_env.apply(dump_video)
        reward = td_test["next", "episode_reward"][td_test["next", "done"]]
        test_rewards.append(reward.cpu())
    del td_test
    return torch.cat(test_rewards, 0).mean()


def make_collector(cfg, train_env, actor_model_explore, compile_mode):
    """Make collector."""
    collector = SyncDataCollector(
        train_env,
        actor_model_explore,
        frames_per_batch=cfg.collector.frames_per_batch,
        total_frames=cfg.collector.total_frames,
        device="cpu",
        compile_policy={"mode": compile_mode} if compile_mode else False,
        cudagraph_policy={"warmup": 10} if cfg.compile.cudagraphs else False,
    )
    #collector.set_seed(cfg.env.seed)
    return collector



def log_metrics(logger, metrics, step):
    for metric_name, metric_value in metrics.items():
        logger.log_scalar(metric_name, metric_value, step)