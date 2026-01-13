"""Utility functions for PPO training with Chronos embeddings."""
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
    VecNormV2,
)

from torchtrade.envs.transforms import CoverageTracker, ChronosEmbeddingTransform
from torchrl.collectors import SyncDataCollector

from torchrl.modules import (
    ActorValueOperator,
    MLP,
    ProbabilisticActor,
    ValueOperator,
    SafeModule,
)

from torchtrade.envs import SeqFuturesSLTPEnv, SeqFuturesSLTPEnvConfig
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

    # Basic OHLCV features for Chronos embedding
    df["features_open"] = df["open"]
    df["features_high"] = df["high"]
    df["features_low"] = df["low"]
    df["features_close"] = df["close"]
    df["features_volume"] = df["volume"]

    # Fill NaN values
    df.fillna(0, inplace=True)

    return df


def env_maker(df, cfg, device="cpu", max_traj_length=1, random_start=False):
    """Create a SeqFuturesSLTPEnv instance."""
    # Convert Hydra ListConfig to regular Python lists
    window_sizes = list(cfg.env.window_sizes)
    stoploss_levels = list(cfg.env.stoploss_levels)
    takeprofit_levels = list(cfg.env.takeprofit_levels)

    config = SeqFuturesSLTPEnvConfig(
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
        stoploss_levels=stoploss_levels,
        takeprofit_levels=takeprofit_levels,
    )
    return SeqFuturesSLTPEnv(df, config, feature_preprocessing_fn=custom_preprocessing)


def apply_env_transforms(env, max_steps, cfg):
    """Apply standard transforms including Chronos embedding to the environment.

    Args:
        env: Base environment
        max_steps: Maximum steps for StepCounter
        cfg: Hydra config containing Chronos settings

    Returns:
        transformed_env: Environment with transforms applied
        vecnorm: The VecNormV2 instance for potential statistics sharing
    """
    # Get market data keys for Chronos embedding
    market_data_keys = [k for k in env.observation_spec.keys() if k.startswith("market_data")]

    # Create Chronos transforms for each market data observation
    chronos_transforms = []
    chronos_out_keys = []

    for market_key in market_data_keys:
        out_key = f"chronos_embedding_{market_key}"
        chronos_out_keys.append(out_key)

        chronos_transform = ChronosEmbeddingTransform(
            in_keys=[market_key],
            out_keys=[out_key],
            model_name=cfg.model.chronos_model,
            aggregation="concat",  # Concatenate embeddings across features
            del_keys=True,  # Remove raw market data after embedding
            device=cfg.optim.device if cfg.optim.device else "cpu",
        )
        chronos_transforms.append(chronos_transform)

    # Get observation keys for normalization (chronos embeddings and account_state)
    # Account state should still be normalized
    obs_keys = ["account_state"]

    vecnorm = VecNormV2(
        in_keys=obs_keys,
        decay=0.99999,
        eps=1e-8,
    )

    # Build transform composition
    transform_list = [
        InitTracker(),
        DoubleToFloat(),
    ]

    # Add all Chronos transforms
    transform_list.extend(chronos_transforms)

    # Add normalization and other transforms
    transform_list.extend([
        vecnorm,
        RewardSum(),
        StepCounter(max_steps=max_steps),
    ])

    transformed_env = TransformedEnv(
        env,
        Compose(*transform_list),
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
    train_env, train_vecnorm = apply_env_transforms(parallel_env, max_train_steps, cfg)

    # Create eval environment with its own VecNormV2
    maker = functools.partial(
        env_maker, test_df, cfg, max_traj_length=max_eval_traj_length
    )
    eval_base_env = ParallelEnv(
        eval_num_envs,
        EnvCreator(maker),
        serial_for_single=True,
    )
    max_eval_steps = test_df.shape[0]

    # Create eval environment with separate VecNormV2
    eval_env, eval_vecnorm = apply_env_transforms(eval_base_env, max_eval_steps, cfg)

    # Create coverage tracker for postproc (used in collector)
    coverage_tracker = CoverageTracker()

    return train_env, eval_env, coverage_tracker


# ====================================================================
# Model utils
# --------------------------------------------------------------------


def make_discrete_ppo_chronos_model(cfg, env, device):
    """Make discrete PPO agent with Chronos embeddings as input."""
    activation = cfg.model.activation
    action_spec = env.action_spec

    # Get chronos embedding keys (these replace market_data keys)
    chronos_keys = [k for k in list(env.observation_spec.keys()) if k.startswith("chronos_embedding")]
    assert len(chronos_keys) > 0, "No chronos embedding keys found in observation spec"

    assert "account_state" in list(env.observation_spec.keys()), "Account state key not in observation spec"
    account_state_key = "account_state"

    # Calculate total input features
    # Each chronos embedding + account state features
    total_input_features = 0
    for key in chronos_keys:
        embedding_shape = env.observation_spec[key].shape
        total_input_features += embedding_shape[-1]

    # Account state: 10 features for futures environment
    account_state_dim = env.observation_spec[account_state_key].shape[-1]
    total_input_features += account_state_dim

    print(f"Total input features (Chronos embeddings + account state): {total_input_features}")
    print(f"  Chronos embedding keys: {chronos_keys}")
    print(f"  Account state dim: {account_state_dim}")

    # Common feature extractor - directly takes concatenated chronos embeddings + account state
    # TorchRL's MLP with multiple in_keys will automatically concatenate them
    common = MLP(
        in_features=total_input_features,
        num_cells=[256, 256, 128],
        out_features=128,
        activation_class=ACTIVATIONS[activation],
        device=device,
    )

    common_module = SafeModule(
        module=common,
        in_keys=chronos_keys + [account_state_key],
        out_keys=["common_features"],
    )

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
    common_module, policy_module, value_module = make_discrete_ppo_chronos_model(
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
        postproc=postproc,
    )
    collector.set_seed(cfg.env.seed)
    return collector


def log_metrics(logger, metrics, step):
    """Log metrics to the logger."""
    for metric_name, metric_value in metrics.items():
        logger.log_scalar(metric_name, metric_value, step)
