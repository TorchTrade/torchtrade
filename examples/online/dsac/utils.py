from __future__ import annotations

import functools
import tempfile
from contextlib import nullcontext

import torch
from tensordict.nn import InteractionType

from torch import optim
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

from torchrl.modules.tensordict_module.actors import ProbabilisticActor
from torchrl.objectives import SoftUpdate
from torchrl.objectives.sac import DiscreteSACLoss
from torchtrade.envs.offline.seqlongonly import SeqLongOnlyEnv, SeqLongOnlyEnvConfig
from torchrl.trainers.helpers.models import ACTIVATIONS
from torchtrade.models.simple_encoders import SimpleCNNEncoder
import copy
import ta
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# ====================================================================
# Environment utils
# -----------------


def custom_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess OHLCV dataframe with normalized features for RL trading.

    Expected columns: ["open", "high", "low", "close", "volume"]

    Uses StandardScaler for normalization to avoid VecNormV2 device issues.
    """

    df = df.copy().reset_index(drop=False)

    df["features_close"] = df["close"]
    df["features_open"] = df["open"]
    df["features_high"] = df["high"]
    df["features_low"] = df["low"]
    df["features_volume"] = df["volume"]

    # Normalize features using StandardScaler
    scaler = StandardScaler()
    feature_cols = [col for col in df.columns if col.startswith("features_")]
    df[feature_cols] = scaler.fit_transform(df[feature_cols])

    df.fillna(0, inplace=True)

    return df


def env_maker(df, cfg, device="cpu"):
    config = SeqLongOnlyEnvConfig(
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
    )
    return SeqLongOnlyEnv(df, config, feature_preprocessing_fn=custom_preprocessing)



def apply_env_transforms(env):
    """Apply standard transforms to the environment.

    Args:
        env: Base environment

    Returns:
        transformed_env: Environment with transforms applied

    Note: Normalization is handled in the preprocessing function using StandardScaler
          to avoid VecNormV2 device issues.
    """
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

    # Create train environment
    train_env = apply_env_transforms(parallel_env)

    # Create eval environment
    maker = functools.partial(env_maker, test_df, cfg)
    eval_base_env = ParallelEnv(
        eval_num_envs,
        EnvCreator(maker),
        serial_for_single=True,
    )
    eval_env = apply_env_transforms(eval_base_env)

    return train_env, eval_env



# ====================================================================
# Collector and replay buffer
# ---------------------------


def make_collector(cfg, train_env, actor_model_explore, compile_mode, device="cpu"):
    """Make collector.

    Args:
        device: Device for data collection (default: "cpu", can use "cuda" now that VecNormV2 is removed)
    """
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
    device=device,
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

    encoders = []

    # Get number of features from environment observation spec
    num_features = env.observation_spec[market_data_keys[0]].shape[-1]

    # Build the encoder
    for key, t, w in zip(market_data_keys, time_frames, window_sizes):

        model = SimpleCNNEncoder(input_shape=(w, num_features),
                            output_shape=(1, 14), # if None, the output shape will be the same as the input shape otherwise you have to provide the output shape (out_seq, out_feat)
                            hidden_channels=64,
                            kernel_size=3,
                            activation="relu",
                            final_activation="relu",
                            dropout=0.1)
        encoders.append(SafeModule(
            module=model,
            in_keys=key,
            out_keys=[f"encoding_{t}_{w}"],
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
    ).to(device)

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
        in_keys=[f"encoding_{t}_{w}" for t, w in zip(time_frames, window_sizes)] + ["encoding_account_state"],
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
        in_keys=[f"encoding_{t}_{w}" for t, w in zip(time_frames, window_sizes)] + ["encoding_account_state"],
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

