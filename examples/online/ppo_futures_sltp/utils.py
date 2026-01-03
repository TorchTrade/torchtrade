"""Utility functions for PPO training on SeqFuturesSLTPEnv."""
from __future__ import annotations
import functools

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

from torchtrade.envs import SeqFuturesSLTPEnv, SeqFuturesSLTPEnvConfig
from torchtrade.envs.offline.utils import TimeFrame, TimeFrameUnit, get_timeframe_unit
import numpy as np
import pandas as pd
from torchrl.trainers.helpers.models import ACTIVATIONS

# ====================================================================
# Environment utils
# --------------------------------------------------------------------


def custom_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess OHLCV dataframe with technical indicators for RL trading.

    Expected columns: ["open", "high", "low", "close", "volume"]
    Index can be datetime or integer (or timestamp if from sampler).
    """
    df = df.copy()

    # Handle HuggingFace datasets with numeric column names
    expected_cols = ["timestamp", "open", "high", "low", "close", "volume"]
    if "close" not in df.columns and "0" in df.columns:
        df.columns = expected_cols[:len(df.columns)]

    # Preserve timestamp index if present (from sampler resampling)
    has_datetime_index = isinstance(df.index, pd.DatetimeIndex)
    if has_datetime_index:
        df = df.reset_index()  # Convert timestamp index to column
        df = df.rename(columns={"index": "timestamp"})  # In case it's named 'index'

    # === Normalized OHLCV (relative to close) ===
    # Note: We don't include raw close price as it's unnormalized (~$50k for BTC)
    # The account_state already contains current_price for the network
    df["features_open_rel"] = df["open"] / df["close"]
    df["features_high_rel"] = df["high"] / df["close"]
    df["features_low_rel"] = df["low"] / df["close"]

    # Log volume (normalized)
    df["features_log_volume"] = np.log1p(df["volume"])
    vol_mean = df["features_log_volume"].rolling(20).mean()
    vol_std = df["features_log_volume"].rolling(20).std()
    df["features_volume_zscore"] = (df["features_log_volume"] - vol_mean) / (vol_std + 1e-8)

    # === Returns ===
    df["features_return_1"] = df["close"].pct_change(1)
    df["features_return_5"] = df["close"].pct_change(5)
    df["features_return_15"] = df["close"].pct_change(15)

    # === Moving Averages (relative to price) ===
    for period in [5, 10, 20]:
        ma = df["close"].rolling(period).mean()
        df[f"features_ma{period}_rel"] = df["close"] / ma - 1

    # === RSI ===
    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-8)
    df["features_rsi"] = (100 - (100 / (1 + rs))) / 100 - 0.5  # Normalize to [-0.5, 0.5]

    # === Bollinger Bands ===
    bb_ma = df["close"].rolling(20).mean()
    bb_std = df["close"].rolling(20).std()
    df["features_bb_upper_dist"] = (df["close"] - (bb_ma + 2 * bb_std)) / df["close"]
    df["features_bb_lower_dist"] = (df["close"] - (bb_ma - 2 * bb_std)) / df["close"]
    df["features_bb_width"] = (4 * bb_std) / df["close"]

    # === MACD ===
    ema12 = df["close"].ewm(span=12).mean()
    ema26 = df["close"].ewm(span=26).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9).mean()
    df["features_macd"] = macd / df["close"]  # Normalize
    df["features_macd_signal"] = signal / df["close"]
    df["features_macd_hist"] = (macd - signal) / df["close"]

    # === Volatility ===
    df["features_volatility_5"] = df["features_return_1"].rolling(5).std()
    df["features_volatility_20"] = df["features_return_1"].rolling(20).std()

    # === ATR (Average True Range) ===
    high_low = df["high"] - df["low"]
    high_close = abs(df["high"] - df["close"].shift())
    low_close = abs(df["low"] - df["close"].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["features_atr"] = tr.rolling(14).mean() / df["close"]

    # === Price momentum ===
    df["features_momentum_10"] = df["close"] / df["close"].shift(10) - 1
    df["features_momentum_30"] = df["close"] / df["close"].shift(30) - 1

    # Fill NaN values
    df.fillna(0, inplace=True)

    # Clip extreme values to prevent NaN from outliers
    feature_cols = [c for c in df.columns if c.startswith("features_")]
    for col in feature_cols:
        df[col] = df[col].clip(-10, 10)

    return df


def env_maker(df, cfg, device="cpu", max_traj_length=1, random_start=False):
    """Create a SeqFuturesSLTPEnv instance."""
    # Convert Hydra ListConfig to regular Python lists
    window_sizes = list(cfg.env.window_sizes)
    execute_on = list(cfg.env.execute_on)
    stoploss_levels = list(cfg.env.stoploss_levels)
    takeprofit_levels = list(cfg.env.takeprofit_levels)

    time_frames = [
        TimeFrame(t, get_timeframe_unit(f))
        for t, f in zip(cfg.env.time_frames, cfg.env.freqs)
    ]
    execute_on = TimeFrame(execute_on[0], get_timeframe_unit(execute_on[1]))

    config = SeqFuturesSLTPEnvConfig(
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
        random_start=random_start,
        leverage=cfg.env.leverage,
        stoploss_levels=stoploss_levels,
        takeprofit_levels=takeprofit_levels,
    )
    return SeqFuturesSLTPEnv(df, config, feature_preprocessing_fn=custom_preprocessing)


def apply_env_transforms(env, max_steps):
    """Apply standard transforms to the environment."""
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


def make_environment(
    train_df,
    test_df,
    cfg,
    train_num_envs=1,
    eval_num_envs=1,
    max_train_traj_length=1,
    max_eval_traj_length=1,
):
    """Make environments for training and evaluation."""
    maker = functools.partial(
        env_maker, train_df, cfg, max_traj_length=max_train_traj_length, random_start=True
    )
    max_train_steps = train_df.shape[0]
    parallel_env = ParallelEnv(
        train_num_envs,
        EnvCreator(maker),
        serial_for_single=True,
    )
    parallel_env.set_seed(cfg.env.seed)

    train_env = apply_env_transforms(parallel_env, max_train_steps)

    maker = functools.partial(
        env_maker, test_df, cfg, max_traj_length=max_eval_traj_length
    )
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
# Model utils
# --------------------------------------------------------------------


def make_discrete_ppo_mlp_model(cfg, env, device):
    """Make discrete PPO agent with MLP encoder."""
    activation = cfg.model.activation
    action_spec = env.action_spec
    market_data_keys = [
        k for k in list(env.observation_spec.keys()) if k.startswith("market_data")
    ]
    assert (
        "account_state" in list(env.observation_spec.keys())
    ), "Account state key not in observation spec"

    time_frames = cfg.env.time_frames
    window_sizes = cfg.env.window_sizes
    freqs = cfg.env.freqs
    assert len(time_frames) == len(
        market_data_keys
    ), f"Amount of time frames {len(time_frames)} and env market data keys do not match! Keys: {market_data_keys}"

    # Build simple MLP encoders for each market data input
    encoders = []
    for key, t, w, freq in zip(market_data_keys, time_frames, window_sizes, freqs):
        num_features = env.observation_spec[key].shape[-1]
        input_dim = w * num_features

        # Flatten and encode market data
        # Use simple input clamping instead of LayerNorm to avoid division by near-zero variance
        class SafeInputClamp(torch.nn.Module):
            def forward(self, x):
                # Clamp extreme values - features already preprocessed to reasonable range
                return torch.clamp(x, -10.0, 10.0)

        encoder = SafeModule(
            module=torch.nn.Sequential(
                torch.nn.Flatten(start_dim=-2),
                SafeInputClamp(),
                MLP(
                    in_features=input_dim,
                    out_features=32,
                    num_cells=[64],
                    activation_class=ACTIVATIONS[activation],
                    device=device,
                ),
            ),
            in_keys=[key],
            out_keys=[f"encoding_{t}_{freq}_{w}"],
        ).to(device)
        encoders.append(encoder)

    # Account state encoder (10 elements for futures)
    # Per-feature normalization for account state:
    # [balance, position_size, position_value, entry_price, current_price,
    #  unrealized_pnl_pct, leverage, margin_ratio, liquidation_price, hold_counter]
    # We need to normalize each feature by its expected scale
    class AccountStateNormalizer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            # Expected scales for each feature (order matters!)
            # [balance, pos_size, pos_value, entry_price, current_price,
            #  pnl_pct, leverage, margin_ratio, liq_price, hold_counter]
            self.register_buffer("scales", torch.tensor([
                5000.0,    # balance (~1000-5000) -> normalize to ~0.2-1.0
                1.0,       # position_size (small, keep as-is but clip)
                5000.0,    # position_value (similar to balance)
                100000.0,  # entry_price (~50k for BTC) -> normalize to ~0.5
                100000.0,  # current_price (~50k for BTC) -> normalize to ~0.5
                1.0,       # unrealized_pnl_pct (already small, keep as-is)
                10.0,      # leverage (typically 1-10) -> normalize to ~0.5
                5.0,       # margin_ratio (typically 0-5)
                100000.0,  # liquidation_price (~50k or 0)
                100.0,     # hold_counter (typically 0-100 steps)
            ]))

        def forward(self, x):
            # Divide by scales and clip to prevent extreme values
            normalized = x / self.scales
            return torch.clamp(normalized, -10.0, 10.0)

    account_encoder = SafeModule(
        module=torch.nn.Sequential(
            AccountStateNormalizer(),
            MLP(
                in_features=10,
                out_features=32,
                num_cells=[32],
                activation_class=ACTIVATIONS[activation],
                device=device,
            ),
        ),
        in_keys=["account_state"],
        out_keys=["encoding_account_state"],
    ).to(device)

    # Common feature extractor
    encoding_keys = [
        f"encoding_{t}_{freq}_{w}"
        for t, w, freq in zip(time_frames, window_sizes, freqs)
    ] + ["encoding_account_state"]
    total_encoding_dim = 32 * (len(market_data_keys) + 1)

    # Safe feature module that prevents NaN/Inf
    class SafeFeatures(torch.nn.Module):
        def __init__(self, mlp):
            super().__init__()
            self.mlp = mlp

        def forward(self, *inputs):
            # Concatenate inputs
            x = torch.cat(inputs, dim=-1)
            # Check for NaN in input and replace with zeros
            x = torch.where(torch.isnan(x) | torch.isinf(x), torch.zeros_like(x), x)
            # Clamp inputs
            x = torch.clamp(x, -100.0, 100.0)
            # Run through MLP
            out = self.mlp(x)
            # Check output
            out = torch.where(torch.isnan(out) | torch.isinf(out), torch.zeros_like(out), out)
            return out

    common = MLP(
        in_features=total_encoding_dim,
        num_cells=[128, 128],
        out_features=128,
        activation_class=ACTIVATIONS[activation],
        device=device,
    )

    common_module = SafeModule(
        module=SafeFeatures(common),
        in_keys=encoding_keys,
        out_keys=["common_features"],
    )
    common_module = SafeSequential(*encoders, account_encoder, common_module)

    # Policy head - action_spec.n gives the number of actions
    action_out_features = action_spec.n
    distribution_class = torch.distributions.Categorical
    distribution_kwargs = {}

    # Safe logits module that prevents NaN/Inf from propagating
    class SafeLogits(torch.nn.Module):
        def forward(self, x):
            # Replace any NaN/Inf with zeros (uniform distribution)
            x = torch.where(torch.isnan(x) | torch.isinf(x), torch.zeros_like(x), x)
            # Clamp to reasonable range for softmax stability
            return torch.clamp(x, -20.0, 20.0)

    # Policy head with logit clamping to prevent extreme values
    policy_net = torch.nn.Sequential(
        MLP(
            in_features=128,
            out_features=action_out_features,
            activation_class=ACTIVATIONS[activation],
            num_cells=[],
            device=device,
        ),
        torch.nn.Tanh(),  # Bound logits to [-1, 1]
        SafeLogits(),     # Additional safety layer
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

    # Value head
    value_net = MLP(
        activation_class=torch.nn.ReLU,
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


def init_weights(module, gain=1.0):
    """Initialize weights with orthogonal initialization for stability."""
    if isinstance(module, torch.nn.Linear):
        torch.nn.init.orthogonal_(module.weight, gain=gain)
        if module.bias is not None:
            torch.nn.init.zeros_(module.bias)


def make_ppo_models(env, device, cfg):
    """Create PPO actor and critic models."""
    common_module, policy_module, value_module = make_discrete_ppo_mlp_model(
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

    # Apply orthogonal weight initialization for training stability
    actor_critic.apply(lambda m: init_weights(m, gain=np.sqrt(2)))
    # Use smaller gain for policy output layer to start with near-uniform action probs
    for module in actor_critic.modules():
        if isinstance(module, torch.nn.Linear):
            if module.out_features == env.action_spec.n:
                init_weights(module, gain=0.01)

    with torch.no_grad():
        td = env.fake_tensordict().unsqueeze(0).expand(3, 2).to(actor_critic.device)
        actor_critic(td)
        del td

    total_params = sum(p.numel() for p in actor_critic.parameters())
    print(f"Total number of parameters: {total_params}")

    actor = actor_critic.get_policy_operator()
    critic = actor_critic.get_value_operator()

    return actor, critic


def make_collector(cfg, train_env, actor_model_explore, compile_mode):
    """Make data collector."""
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
        compile_policy={"mode": compile_mode} if compile_mode else False,
        cudagraph_policy={"warmup": 10} if cfg.compile.cudagraphs else False,
    )
    collector.set_seed(cfg.env.seed)
    return collector


def log_metrics(logger, metrics, step):
    """Log metrics to the logger."""
    for metric_name, metric_value in metrics.items():
        logger.log_scalar(metric_name, metric_value, step)
