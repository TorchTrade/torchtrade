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
from torchtrade.models.simple_encoders import SimpleCNNEncoder, SimpleMLPEncoder

from torchtrade.envs import SeqFuturesSLTPEnv, SeqFuturesSLTPEnvConfig
from torchtrade.envs import FuturesOneStepEnv, FuturesOneStepEnvConfig
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

    # Basic OHLCV features (same as PPO)
    df["features_close"] = df["close"]
    df["features_open"] = df["open"]
    df["features_high"] = df["high"]
    df["features_low"] = df["low"]
    df["features_volume"] = df["volume"]

    # Fill NaN values
    df.fillna(0, inplace=True)

    return df


def env_maker(df, cfg, device="cpu", max_traj_length=1, eval=False):
    """Create a FuturesOneStepEnv or SeqFuturesSLTPEnv instance.

    Both environments share the same 19-action space:
    - Action 0: Hold/Close
    - Actions 1-9: Long positions with SL/TP combinations
    - Actions 10-18: Short positions with SL/TP combinations

    Training uses FuturesOneStepEnv (one-step GRPO-style training).
    Evaluation uses SeqFuturesSLTPEnv (sequential episodes with render_history).
    """
    # Convert Hydra ListConfig to regular Python lists
    window_sizes = list(cfg.env.window_sizes)
    stoploss_levels = list(cfg.env.stoploss_levels)
    takeprofit_levels = list(cfg.env.takeprofit_levels)

    if not eval:
        # Training: use FuturesOneStepEnv for GRPO-style training
        config = FuturesOneStepEnvConfig(
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
            leverage=cfg.env.leverage,
            stoploss_levels=stoploss_levels,
            takeprofit_levels=takeprofit_levels,
            max_traj_length=max_traj_length,
        )
        return FuturesOneStepEnv(df, config, feature_preprocessing_fn=custom_preprocessing)
    else:
        # Evaluation: use SeqFuturesSLTPEnv for sequential evaluation
        # Both envs have the same 19-action space (1 hold + 9 long + 9 short)
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
            leverage=cfg.env.leverage,
            stoploss_levels=stoploss_levels,
            takeprofit_levels=takeprofit_levels,
            max_traj_length=max_traj_length,
            random_start=False,
        )
        return SeqFuturesSLTPEnv(df, config, feature_preprocessing_fn=custom_preprocessing)


def apply_env_transforms(env, max_steps, one_step_env=True):
    """Apply standard transforms to the environment.

    Args:
        env: The environment to transform
        max_steps: Maximum steps per episode
        one_step_env: If True, skip unnecessary transforms for one-step envs

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

    if one_step_env:
        transformed_env = TransformedEnv(
            env,
            Compose(
                InitTracker(),
                vecnorm,
                StepCounter(max_steps=max_steps),
            ),
        )
    else:
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
    train_env, train_vecnorm = apply_env_transforms(parallel_env, max_train_steps, one_step_env=True)

    maker = functools.partial(
        env_maker, test_df, cfg, max_traj_length=max_eval_traj_length, eval=True
    )
    # Eval env uses SeqFuturesSLTPEnv which is NOT one-step, so use full transforms
    eval_parallel_env = ParallelEnv(
        eval_num_envs,
        EnvCreator(maker),
        serial_for_single=True,
    )
    eval_env, eval_vecnorm = apply_env_transforms(eval_parallel_env, max_eval_traj_length, one_step_env=False)

    # Create coverage tracker for postproc (used in collector)
    # Initialize it properly by setting the parent to access base env
    coverage_tracker = CoverageTracker()
    coverage_tracker.parent = train_env

    # Force initialization by creating a fake reset
    from tensordict import TensorDict
    fake_reset = train_env.reset()
    coverage_tracker._reset(TensorDict({}), fake_reset)

    print(f"CoverageTracker initialized with {coverage_tracker._num_positions} positions")

    return train_env, eval_env, coverage_tracker



# ====================================================================
# Model utils
# --------------------------------------------------------------------
# NOTE: Safety wrappers (AccountStateNormalizer, SafeEncoderWrapper, etc.)
# were removed after confirming BiNMTABL works fine with clean datasets.
# NaN issues were caused by data gaps, not model architecture.
# If you see NaN during training, check your dataset for missing periods.


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


def make_discrete_ppo_binmtabl_model(cfg, env, device):
    """Make discrete PPO agent with BiNMTABL encoder."""
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
        model = SimpleCNNEncoder(
            input_shape=(w, num_features),
            output_shape=(1, 14),
            hidden_channels=64,
            kernel_size=3,
            activation=activation,
            final_activation=activation,
            dropout=0.1
        )
        encoders.append(SafeModule(
            module=model,
            in_keys=key,
            out_keys=[f"encoding_{t}_{w}"],
        ).to(device))

    # Account state encoder with SimpleMLPEncoder
    # IMPORTANT: SeqFuturesSLTPEnv has 10 account state features (not 7 like SeqLongOnly)
    # [cash, position_size, position_value, entry_price, current_price,
    #  unrealized_pnl_pct, leverage, margin_ratio, liquidation_price, holding_time]
    account_encoder_model = SimpleMLPEncoder(
        input_shape=(1, 10),  # 10 features for futures (vs 7 for long-only)
        output_shape=(1, 14),  # Match embedding_dim output
        hidden_sizes=(32, 32),
        activation="gelu",
        dropout=0.1,
        final_activation="gelu",
    )

    account_state_encoder = SafeModule(
        module=account_encoder_model,
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
    """Create PPO actor and critic models.

    Returns:
        encoder: The common encoder module (for CTRL loss)
        actor: The policy operator
        critic: The value operator
    """
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
    encoder = common_module  # Return encoder for CTRL loss

    return encoder, actor, critic


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
