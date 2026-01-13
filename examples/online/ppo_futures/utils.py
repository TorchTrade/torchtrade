"""Utility functions for PPO training on SeqFuturesEnv."""
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
    VecNormV2,
)

from torchtrade.envs.transforms import CoverageTracker
from torchrl.collectors import SyncDataCollector

from torchrl.modules import (
    ActorValueOperator,
    MLP,
    ProbabilisticActor,
    ValueOperator,
    SafeModule,
    SafeSequential,
)
from torchtrade.models import SimpleCNNEncoder

from torchtrade.envs import SeqFuturesEnv, SeqFuturesEnvConfig
import numpy as np
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


def env_maker(df, cfg, device="cpu", max_traj_length=1, random_start=False):
    """Create a SeqFuturesEnv instance."""
    # Convert Hydra ListConfig to regular Python lists
    window_sizes = list(cfg.env.window_sizes)

    config = SeqFuturesEnvConfig(
        symbol=cfg.env.symbol,
        time_frames=cfg.env.time_frames,
        window_sizes=window_sizes,
        execute_on=cfg.env.execute_on,
        include_base_features=False,
        initial_cash=cfg.env.initial_cash,
        slippage=cfg.env.slippage,
        transaction_fee=cfg.env.transaction_fee,
        bankrupt_threshold=cfg.env.bankrupt_threshold,
        seed=cfg.env.seed,
        max_traj_length=max_traj_length,
        random_start=random_start,
        leverage=cfg.env.leverage,
    )
    return SeqFuturesEnv(df, config, feature_preprocessing_fn=custom_preprocessing)


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

    # Create train environment and get its VecNormV2 instance
    train_env, train_vecnorm = apply_env_transforms(parallel_env, max_train_steps)

    # Create eval environment with its own VecNormV2
    # Note: Each VecNormV2 will maintain its own running statistics
    # This is acceptable as eval will normalize observations consistently during evaluation
    maker = functools.partial(
        env_maker, test_df, cfg, max_traj_length=max_eval_traj_length
    )
    eval_base_env = ParallelEnv(
        eval_num_envs,
        EnvCreator(maker),
        serial_for_single=True,
    )
    max_eval_steps = test_df.shape[0]

    # Create eval environment with separate VecNormV2 (will compute its own statistics)
    eval_env, eval_vecnorm = apply_env_transforms(eval_base_env, max_eval_steps)

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


def make_discrete_ppo_model(cfg, env, device):
    """Make discrete PPO agent"""
    activation = cfg.model.activation
    action_spec = env.action_spec
    # TODO: should use the get_market_data function of the env
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
    # IMPORTANT: SeqFuturesEnv has 10 account state features (not 7 like SeqLongOnly)
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

    action_out_features = action_spec.n
    distribution_class = torch.distributions.Categorical
    distribution_kwargs = {}

    # Policy head
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


def make_ppo_models(env, device, cfg):
    """Create PPO actor and critic models."""
    common_module, policy_module, value_module = make_discrete_ppo_model(
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


def make_collector(cfg, train_env, actor_model_explore, compile_mode, postproc=None):
    """Make data collector."""
    collector = SyncDataCollector(
        train_env,
        actor_model_explore,
        frames_per_batch=cfg.collector.frames_per_batch,
        total_frames=cfg.collector.total_frames,
        device="cpu",
        compile_policy={"mode": compile_mode} if compile_mode else False,
        cudagraph_policy={"warmup": 10} if cfg.compile.cudagraphs else False,
        postproc=postproc,  # Add coverage tracker as postproc
    )
    collector.set_seed(cfg.env.seed)
    return collector


def log_metrics(logger, metrics, step):
    """Log metrics to the logger."""
    for metric_name, metric_value in metrics.items():
        logger.log_scalar(metric_name, metric_value, step)
