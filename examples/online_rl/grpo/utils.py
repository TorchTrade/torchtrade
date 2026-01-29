"""Utility functions for GRPO training on OneStepTradingEnv."""
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

from torchtrade.envs.transforms import CoverageTracker
from torchrl.collectors import SyncDataCollector

from torchrl.modules import (
    MLP,
    ProbabilisticActor,
    SafeModule,
    SafeSequential,
)
from torchtrade.models import SimpleCNNEncoder

from torchtrade.envs import (
    OneStepTradingEnv,
    OneStepTradingEnvConfig,
    SequentialTradingEnvSLTP,
    SequentialTradingEnvSLTPConfig,
)
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torchrl.trainers.helpers.models import ACTIVATIONS

# ====================================================================
# Environment utils
# --------------------------------------------------------------------


def custom_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess OHLCV dataframe with normalized features for RL trading.

    Expected columns: ["open", "high", "low", "close", "volume"]
    Index can be datetime or integer.

    Uses StandardScaler for normalization to avoid VecNormV2 device issues.
    """
    df = df.copy().reset_index(drop=False)

    # Basic OHLCV features (same as PPO)
    df["features_close"] = df["close"]
    df["features_open"] = df["open"]
    df["features_high"] = df["high"]
    df["features_low"] = df["low"]
    df["features_volume"] = df["volume"]

    # Normalize features using StandardScaler
    scaler = StandardScaler()
    feature_cols = [col for col in df.columns if col.startswith("features_")]
    df[feature_cols] = scaler.fit_transform(df[feature_cols])

    # Fill NaN values
    df.fillna(0, inplace=True)

    return df


def env_maker(df, cfg, device="cpu", max_traj_length=1, eval=False):
    """Create a OneStepTradingEnv or SequentialTradingEnvSLTP instance.

    Both environments share the same 19-action space:
    - Action 0: Hold/Close
    - Actions 1-9: Long positions with SL/TP combinations
    - Actions 10-18: Short positions with SL/TP combinations

    Training uses OneStepTradingEnv (one-step GRPO-style training).
    Evaluation uses SequentialTradingEnvSLTP (sequential episodes with render_history).
    """
    if not eval:
        # Training: use OneStepTradingEnv for GRPO-style training
        config = OneStepTradingEnvConfig(
            symbol=cfg.env.symbol,
            time_frames=cfg.env.time_frames,
            window_sizes=cfg.env.window_sizes,
            execute_on=cfg.env.execute_on,
            include_base_features=False,
            initial_cash=cfg.env.initial_cash,
            slippage=cfg.env.slippage,
            transaction_fee=cfg.env.transaction_fee,
            bankrupt_threshold=cfg.env.bankrupt_threshold,
            seed=cfg.env.seed,
            leverage=cfg.env.leverage,
            stoploss_levels=cfg.env.stoploss_levels,
            takeprofit_levels=cfg.env.takeprofit_levels,
            max_traj_length=max_traj_length,
            include_hold_action=cfg.env.include_hold_action,
        )
        return OneStepTradingEnv(df, config, feature_preprocessing_fn=custom_preprocessing)
    else:
        # Evaluation: use SequentialTradingEnvSLTP for sequential evaluation
        # Both envs have the same 19-action space (1 hold + 9 long + 9 short)
        config = SequentialTradingEnvSLTPConfig(
            symbol=cfg.env.symbol,
            time_frames=cfg.env.time_frames,
            window_sizes=cfg.env.window_sizes,
            execute_on=cfg.env.execute_on,
            include_base_features=False,
            initial_cash=cfg.env.initial_cash,
            slippage=cfg.env.slippage,
            transaction_fee=cfg.env.transaction_fee,
            bankrupt_threshold=cfg.env.bankrupt_threshold,
            seed=cfg.env.seed,
            leverage=cfg.env.leverage,
            stoploss_levels=cfg.env.stoploss_levels,
            takeprofit_levels=cfg.env.takeprofit_levels,
            max_traj_length=max_traj_length,
            random_start=False,
        )
        return SequentialTradingEnvSLTP(df, config, feature_preprocessing_fn=custom_preprocessing)


def apply_env_transforms(env, max_steps, one_step_env=True):
    """Apply standard transforms to the environment.

    Args:
        env: The environment to transform
        max_steps: Maximum steps per episode
        one_step_env: If True, skip unnecessary transforms for one-step envs

    Returns:
        transformed_env: Environment with transforms applied

    Note: Normalization is handled in the preprocessing function using StandardScaler
          to avoid VecNormV2 device issues.
    """
    if one_step_env:
        # PERF: Minimal transforms for one-step environments
        # - RewardSum is unnecessary (each step terminates)
        # - DoubleToFloat skipped (data already float32 from environment)
        transformed_env = TransformedEnv(
            env,
            Compose(
                InitTracker(),
                StepCounter(max_steps=max_steps),
            ),
        )
    else:
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

    # PERF: Use minimal transforms for one-step training env
    train_env = apply_env_transforms(parallel_env, max_train_steps, one_step_env=True)

    maker = functools.partial(
        env_maker, test_df, cfg, max_traj_length=max_eval_traj_length, eval=True
    )
    # Eval env uses SequentialTradingEnvSLTP which is NOT one-step, so use full transforms
    eval_parallel_env = ParallelEnv(
        eval_num_envs,
        EnvCreator(maker),
        serial_for_single=True,
    )
    eval_env = apply_env_transforms(eval_parallel_env, max_eval_traj_length, one_step_env=False)

    # Create coverage tracker for postproc (used in collector)
    coverage_tracker = CoverageTracker()

    return train_env, eval_env, coverage_tracker


# ====================================================================
# Model utils
# --------------------------------------------------------------------


class BatchSafeWrapper(torch.nn.Module):
    """Wrapper to ensure consistent batch dimension in output."""
    def __init__(self, model, output_features):
        super().__init__()
        self.model = model
        self.output_features = output_features

    def forward(self, x):
        out = self.model(x)
        # If batch dimension was squeezed (batch_size=1), add it back
        if out.dim() == 1 and out.shape[0] == self.output_features:
            out = out.unsqueeze(0)
        return out


def make_discrete_grpo_model(cfg, env, device):
    """Make discrete GRPO agent encoder (same as PPO)."""
    activation = "tanh"
    action_spec = env.action_spec
    market_data_keys = [k for k in list(env.observation_spec.keys()) if k.startswith("market_data")]
    assert "account_state" in list(env.observation_spec.keys()), "Account state key not in observation spec"
    account_state_key = "account_state"

    time_frames = cfg.env.time_frames
    window_sizes = cfg.env.window_sizes

    encoders = []
    num_features = env.observation_spec[market_data_keys[0]].shape[-1]

    # Build CNN encoders for market data
    for key, t, w in zip(market_data_keys, time_frames, window_sizes):
        base_model = SimpleCNNEncoder(
            input_shape=(w, num_features),
            output_shape=(1, 14),
            hidden_channels=64,
            kernel_size=3,
            activation=activation,
            final_activation=activation,
            dropout=0.1,
        )
        # Wrap to handle batch_size=1 case where encoder might squeeze batch dim
        model = BatchSafeWrapper(base_model, output_features=14)
        encoders.append(SafeModule(
            module=model,
            in_keys=key,
            out_keys=[f"encoding_{t}_{w}"],
        ).to(device))

    # Account state encoder with MLP
    # IMPORTANT: OneStepTradingEnv has 10 account state features (not 7 like SeqLongOnly)
    # [cash, position_size, position_value, entry_price, current_price,
    #  unrealized_pnl_pct, leverage, margin_ratio, liquidation_price, holding_time]
    account_state_encoder = SafeModule(
        module=MLP(
            num_cells=[32, 32],
            out_features=14,
            activation_class=ACTIVATIONS[activation],
            device=device,
        ),
        in_keys=[account_state_key],
        out_keys=["encoding_account_state"],
    ).to(device)

    # Common feature extractor
    common = MLP(
        num_cells=[128, 128],
        out_features=128,
        activation_class=ACTIVATIONS[activation],
        device=device,
    )

    common_module = SafeModule(
        module=common,
        in_keys=[f"encoding_{t}_{w}" for t, w in zip(time_frames, window_sizes)] + ["encoding_account_state"],
        out_keys=["common_features"],
    )
    common_module = SafeSequential(*encoders, account_state_encoder, common_module)

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
    policy = make_discrete_grpo_model(
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


def make_collector(cfg, train_env, actor_model_explore, compile_mode, postproc=None, device="cpu"):
    """Make data collector.

    Args:
        device: Device for data collection (default: "cpu", can use "cuda" now that VecNormV2 is removed)
    """
    collector = SyncDataCollector(
        train_env,
        actor_model_explore,
        frames_per_batch=cfg.collector.frames_per_batch,
        total_frames=cfg.collector.total_frames,
        device=device,
        compile_policy={"mode": compile_mode} if compile_mode else False,
        cudagraph_policy={"warmup": 10} if cfg.compile.cudagraphs else False,
        postproc=postproc,  # Add coverage tracker as postproc
    )
    return collector


def log_metrics(logger, metrics, step):
    """Log metrics to the logger."""
    import torch
    for metric_name, metric_value in metrics.items():
        # PERF: Convert tensors to Python floats for logging (single sync point)
        if isinstance(metric_value, torch.Tensor):
            metric_value = metric_value.item()
        logger.log_scalar(metric_name, metric_value, step)
