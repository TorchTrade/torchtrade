"""Utility functions for GRPO training on FuturesOneStepEnv."""
from __future__ import annotations
import functools

import torch.nn
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
    MLP,
    ProbabilisticActor,
    SafeModule,
    SafeSequential,
)

from torchtrade.envs import (
    FuturesOneStepEnv,
    FuturesOneStepEnvConfig,
)
from torchtrade.envs.offline.utils import TimeFrame, TimeFrameUnit, get_timeframe_unit
import pandas as pd
from torchrl.trainers.helpers.models import ACTIVATIONS

# ====================================================================
# Environment utils
# --------------------------------------------------------------------


def custom_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess OHLCV dataframe with features for RL trading.

    Expected columns: ["open", "high", "low", "close", "volume"]
    Index can be datetime or integer.
    """
    df = df.copy().reset_index(drop=False)

    # Basic OHLCV features
    df["features_close"] = df["close"]
    df["features_open"] = df["open"]
    df["features_high"] = df["high"]
    df["features_low"] = df["low"]
    df["features_volume"] = df["volume"]

    # Fill NaN values
    df.fillna(0, inplace=True)

    return df


def env_maker(df, cfg, device="cpu", max_traj_length=1, eval=False):
    """Create a FuturesOneStepEnv instance.

    Note: We use FuturesOneStepEnv for both training and evaluation to ensure
    consistent action space (19 actions with SL/TP combinations).
    """
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

    # Use FuturesOneStepEnv for both training and eval to maintain
    # consistent 19-action space (1 hold + 9 long SL/TP + 9 short SL/TP)
    config = FuturesOneStepEnvConfig(
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
        leverage=cfg.env.leverage,
        stoploss_levels=stoploss_levels,
        takeprofit_levels=takeprofit_levels,
        max_traj_length=max_traj_length,
    )
    return FuturesOneStepEnv(df, config, feature_preprocessing_fn=custom_preprocessing)


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
        env_maker, train_df, cfg, max_traj_length=max_train_traj_length
    )
    max_train_steps = train_df.shape[0]
    parallel_env = ParallelEnv(
        train_num_envs,
        EnvCreator(maker),
        serial_for_single=True,
    )
    parallel_env.set_seed(cfg.env.seed, static_seed=True)  # needs to be static for GRPO style training

    train_env = apply_env_transforms(parallel_env, max_train_steps)

    maker = functools.partial(
        env_maker, test_df, cfg, max_traj_length=max_eval_traj_length, eval=True
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


def make_discrete_grpo_mlp_model(cfg, env, device):
    """Make discrete GRPO agent with MLP encoder."""
    activation = "tanh"
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
        encoder = SafeModule(
            module=torch.nn.Sequential(
                torch.nn.Flatten(start_dim=-2),
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
    account_encoder = SafeModule(
        module=MLP(
            in_features=10,
            out_features=32,
            num_cells=[32],
            activation_class=ACTIVATIONS[activation],
            device=device,
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

    common = MLP(
        in_features=total_encoding_dim,
        num_cells=[128, 128],
        out_features=128,
        activation_class=ACTIVATIONS[activation],
        device=device,
    )

    common_module = SafeModule(
        module=common,
        in_keys=encoding_keys,
        out_keys=["common_features"],
    )
    common_module = SafeSequential(*encoders, account_encoder, common_module)

    # Policy head
    action_out_features = action_spec.n
    distribution_class = torch.distributions.Categorical
    distribution_kwargs = {}

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

    policy_module = SafeSequential(common_module, policy_module)

    policy = ProbabilisticActor(
        policy_module,
        in_keys=["logits"],
        spec=env.full_action_spec_unbatched.to(device),
        distribution_class=distribution_class,
        distribution_kwargs=distribution_kwargs,
        return_log_prob=True,
        default_interaction_type=ExplorationType.RANDOM,
    )

    return policy


def make_grpo_policy(env, device, cfg):
    """Create GRPO policy."""
    policy = make_discrete_grpo_mlp_model(
        cfg,
        env,
        device=device,
    )

    policy.to(device)

    with torch.no_grad():
        td = env.fake_tensordict().unsqueeze(0).expand(3, 2).to(policy.device)
        policy(td)
        del td

    total_params = sum(p.numel() for p in policy.parameters())
    print(f"Total number of parameters: {total_params}")

    return policy


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
    return collector


def log_metrics(logger, metrics, step):
    """Log metrics to the logger."""
    for metric_name, metric_value in metrics.items():
        logger.log_scalar(metric_name, metric_value, step)
