"""Utility functions for DQN training on SequentialTradingEnv."""
from __future__ import annotations
import functools

import torch.nn
from torchrl.envs import (
    DoubleToFloat,
    EnvCreator,
    FlattenObservation,
    ParallelEnv,
    RewardSum,
    InitTracker,
    Compose,
    TransformedEnv,
    StepCounter,
)

from torchtrade.envs.transforms import CoverageTracker
from torchrl.collectors import SyncDataCollector
from torchrl.data import LazyTensorStorage, TensorDictReplayBuffer

from torchrl.modules import (
    MLP,
    SafeModule,
    SafeSequential,
    EGreedyModule,
    QValueActor,
)
from torchtrade.models import SimpleCNNEncoder

from torchtrade.envs import (
    SequentialTradingEnv,
    SequentialTradingEnvConfig,
    SequentialTradingEnvSLTP,
    SequentialTradingEnvSLTPConfig,
)
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torchrl.trainers.helpers.models import ACTIVATIONS
from tensordict.nn import TensorDictModule, TensorDictSequential
import torch.nn as nn

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

    # Basic OHLCV features
    df["features_open"] = df["open"]
    df["features_high"] = df["high"]
    df["features_low"] = df["low"]
    df["features_close"] = df["close"]
    df["features_volume"] = df["volume"]

    # Normalize features using StandardScaler
    scaler = StandardScaler()
    feature_cols = [col for col in df.columns if col.startswith("features_")]
    df[feature_cols] = scaler.fit_transform(df[feature_cols])

    # Fill NaN values
    df.fillna(0, inplace=True)

    return df


def env_maker(df, cfg, device="cpu", max_traj_length=1, random_start=False):
    """Create environment instance based on cfg.env.name."""
    if cfg.env.name == "SequentialTradingEnv":
        config = SequentialTradingEnvConfig(
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
            action_levels=cfg.env.action_levels,
            max_traj_length=max_traj_length,
            random_start=random_start,
        )
        return SequentialTradingEnv(df, config, feature_preprocessing_fn=custom_preprocessing)
    elif cfg.env.name == "SequentialTradingEnvSLTP":
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
            include_hold_action=cfg.env.include_hold_action,
            max_traj_length=max_traj_length,
            random_start=random_start,
        )
        return SequentialTradingEnvSLTP(df, config, feature_preprocessing_fn=custom_preprocessing)
    else:
        raise ValueError(f"Unknown environment: {cfg.env.name}")


def apply_env_transforms(env, max_steps):
    """Apply standard transforms to the environment.

    Args:
        env: Base environment
        max_steps: Maximum steps for StepCounter

    Returns:
        transformed_env: Environment with transforms applied

    Note: Normalization is handled in the preprocessing function using StandardScaler
          to avoid VecNormV2 device issues.
    """
    obs_keys = list(env.observation_spec.keys())
    market_key = [key for key in obs_keys if key.startswith("market_")][0]
    assert type(market_key) == str, "For the TDQN example we only process single time frame observations"
    transformed_env = TransformedEnv(
        env,
        Compose(
            InitTracker(),
            DoubleToFloat(),
            RewardSum(),
            StepCounter(max_steps=max_steps),
            FlattenObservation(in_keys=[market_key], first_dim=-2, last_dim=-1),
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

    # Create train environment
    train_env = apply_env_transforms(parallel_env, max_train_steps)

    # Create eval environment
    maker = functools.partial(
        env_maker, test_df, cfg, max_traj_length=max_eval_traj_length
    )
    eval_base_env = ParallelEnv(
        eval_num_envs,
        EnvCreator(maker),
        serial_for_single=True,
    )
    max_eval_steps = test_df.shape[0]
    eval_env = apply_env_transforms(eval_base_env, max_eval_steps)

    # Create coverage tracker for postproc (used in collector)
    coverage_tracker = CoverageTracker()

    return train_env, eval_env, coverage_tracker


# ====================================================================
# Collector and replay buffer
# --------------------------------------------------------------------


def make_collector(cfg, train_env, policy, compile_mode=None, postproc=None):
    """Make data collector."""
    device = cfg.collector.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

    collector = SyncDataCollector(
        train_env,
        policy,
        frames_per_batch=cfg.collector.frames_per_batch,
        total_frames=cfg.collector.total_frames,
        device=device,
        init_random_frames=cfg.collector.init_random_frames,
        compile_policy={"mode": compile_mode} if compile_mode else False,
        cudagraph_policy={"warmup": 10} if cfg.compile.cudagraphs else False,
        postproc=postproc,
    )
    collector.set_seed(cfg.env.seed)
    return collector


def make_replay_buffer(cfg, device="cpu"):
    """Make replay buffer."""
    storage = LazyTensorStorage(cfg.buffer.buffer_size, device=device)

    return TensorDictReplayBuffer(
        storage=storage,
        batch_size=cfg.buffer.batch_size,
        prefetch=3,
    )


# ====================================================================
# Model
# --------------------------------------------------------------------


class TDQN(nn.Module):
    """Trading DQN Network - Position-aware Q-network for trading.

    Takes two inputs:
    - observation: market data features
    - position: current trading position

    Architecture:
    - 5 fully connected layers with BatchNorm and Dropout
    - Concatenates observation + position as input
    - LeakyReLU activation
    """
    def __init__(
        self,
        input_shape,
        action_dim,
        hidden_size=128,
        dropout=0.2,
        activation=nn.LeakyReLU,
    ):
        super(TDQN, self).__init__()

        self.hidden_size = hidden_size
        self.dropout = dropout
        self.activation = activation()
        self.output_activation = nn.Identity()

        self.fc1 = nn.Linear(input_shape, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, hidden_size)
        self.fc5 = nn.Linear(hidden_size, action_dim)

        # BatchNorm layers
        self.norm1 = nn.BatchNorm1d(hidden_size)
        self.norm2 = nn.BatchNorm1d(hidden_size)
        self.norm3 = nn.BatchNorm1d(hidden_size)
        self.norm4 = nn.BatchNorm1d(hidden_size)

        # Dropout layers
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.dropout4 = nn.Dropout(dropout)

    def forward(self, observation, position):
        """Forward pass with concatenated observation and position."""
        # Concatenate observation and position
        x = torch.cat([observation, position], dim=-1)

        x = self.dropout1(self.activation(self.norm1(self.fc1(x))))
        x = self.dropout2(self.activation(self.norm2(self.fc2(x))))
        x = self.dropout3(self.activation(self.norm3(self.fc3(x))))
        x = self.dropout4(self.activation(self.norm4(self.fc4(x))))
        output = self.output_activation(self.fc5(x))
        return output


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


def make_dqn_agent(cfg, train_env, device):
    """Make DQN agent with epsilon-greedy exploration."""
    activation = cfg.model.activation
    action_spec = train_env.action_spec

    # Get market data keys and account state key
    market_data_keys = [k for k in list(train_env.observation_spec.keys()) if k.startswith("market_data")]
    assert "account_state" in list(train_env.observation_spec.keys()), "Account state key not in observation spec"
    account_state_key = "account_state"

    time_frames = cfg.env.time_frames
    window_sizes = cfg.env.window_sizes

    encoders = []
    num_features = train_env.observation_spec[market_data_keys[0]].shape[-1]

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
    # Account state features are determined dynamically from environment spec
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

    encoder = SafeSequential(*encoders, account_state_encoder)

    # Q-value network - outputs Q-values for all actions
    qvalue_net = MLP(
        num_cells=[128, 128],
        out_features=action_spec.space.n,
        activation_class=ACTIVATIONS[activation],
        device=device,
    )

    qvalue_module = SafeModule(
        module=qvalue_net,
        in_keys=[f"encoding_{t}_{w}" for t, w in zip(time_frames, window_sizes)] + ["encoding_account_state"],
        out_keys=["action_value"],
    )

    # Full Q-value network (encoder + Q-head)
    qvalue_network = SafeSequential(encoder, qvalue_module)

    # Wrap with QValueActor for action selection
    actor = QValueActor(
        module=qvalue_network,
        spec=action_spec,
        action_space="categorical",
    )

    # Add epsilon-greedy exploration
    exploration_module = EGreedyModule(
        spec=action_spec,
        annealing_num_steps=cfg.collector.annealing_frames,
        eps_init=cfg.collector.eps_start,
        eps_end=cfg.collector.eps_end,
    )

    exploration_policy = TensorDictSequential(actor, exploration_module)

    # Test forward pass
    with torch.no_grad():
        td = train_env.fake_tensordict().to(device)
        actor(td)
        del td

    total_params = sum(p.numel() for p in actor.parameters())
    print(f"Total number of parameters: {total_params}")

    return actor, exploration_policy


def make_tdqn_agent(cfg, train_env, device):
    """Make TDQN (Trading DQN) agent.

    Simple approach:
    1. Flatten all observations (market_data + account_state including position)
    2. Pass to deep MLP with BatchNorm and Dropout
    """
    action_spec = train_env.action_spec

    # Get market data keys and account state key
    market_data_keys = [k for k in list(train_env.observation_spec.keys()) if k.startswith("market_data")]
    assert "account_state" in list(train_env.observation_spec.keys()), "Account state key not in observation spec"
    account_state_key = "account_state"

    net = TDQN(
        input_shape=train_env.observation_spec_unbatched[market_data_keys[0]].shape[0] + train_env.observation_spec_unbatched[account_state_key].shape[0],
        action_dim=train_env.action_spec_unbatched.n,
        hidden_size=128,
        dropout=cfg.model.dropout,
    )

    qvalue_module = TensorDictModule(
        net,
        in_keys=market_data_keys + [account_state_key],
        out_keys=["action_value"],
    )

    # Wrap with QValueActor for action selection
    actor = QValueActor(
        module=qvalue_module,
        spec=action_spec,
        action_space="categorical",
    ).to(device)

    # Add epsilon-greedy exploration
    exploration_module = EGreedyModule(
        spec=action_spec,
        annealing_num_steps=cfg.collector.annealing_frames,
        eps_init=cfg.collector.eps_start,
        eps_end=cfg.collector.eps_end,
    )

    exploration_policy = TensorDictSequential(actor, exploration_module).to(device)

    # Test forward pass
    with torch.no_grad():
        td = train_env.fake_tensordict().to(device)
        actor(td)
        del td

    total_params = sum(p.numel() for p in actor.parameters())
    print(f"Total number of parameters: {total_params}")

    return actor, exploration_policy


# ====================================================================
# Loss
# --------------------------------------------------------------------


def make_loss_module(cfg, model):
    """Make DQN loss module and target updater."""
    from torchrl.objectives import DQNLoss
    from torchrl.objectives import HardUpdate

    loss_module = DQNLoss(
        model,
        action_space="categorical",
        delay_value=True,
        loss_function="l2",
    )
    loss_module.make_value_estimator(gamma=cfg.loss.gamma)

    # Use hard target network updates
    target_updater = HardUpdate(
        loss_module,
        value_network_update_interval=cfg.loss.hard_update_freq,
    )

    return loss_module, target_updater


# ====================================================================
# Optimizer
# --------------------------------------------------------------------


def make_optimizer(cfg, loss_module):
    """Make optimizer."""
    import torch.optim

    return torch.optim.Adam(
        loss_module.parameters(),
        lr=cfg.optim.lr,
        eps=1e-8,
    )


# ====================================================================
# Evaluation utils
# --------------------------------------------------------------------


def eval_model(actor, test_env, num_episodes=1, max_steps=10000):
    """Evaluate policy on test environment."""
    test_rewards = []
    for _ in range(num_episodes):
        td = test_env.rollout(
            policy=actor,
            auto_reset=True,
            auto_cast_to_device=False,
            break_when_any_done=True,
            max_steps=max_steps,
        )
        reward = td["next", "episode_reward"][td["next", "done"]]
        test_rewards.append(reward.cpu())
    return torch.cat(test_rewards, 0).mean()


# ====================================================================
# General utils
# --------------------------------------------------------------------


def log_metrics(logger, metrics, step):
    """Log metrics to logger."""
    for key, value in metrics.items():
        logger.log_scalar(key, value, step)
