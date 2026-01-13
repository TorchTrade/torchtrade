from __future__ import annotations

import functools

import torch.nn
import torch.optim
import tensordict
from tensordict.nn import InteractionType
from torch.distributions import Categorical
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
    VecNormV2,
)

from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.modules import (
    MLP,
    ProbabilisticActor,
    SafeModule,
    SafeSequential,
)
from torchrl.objectives import DiscreteIQLLoss, HardUpdate
from torchrl.trainers.helpers.models import ACTIVATIONS
from torchtrade.models import SimpleCNNEncoder

import copy
import pandas as pd
import numpy as np
import ta
from torchtrade.envs.offline.seqlongonly import SeqLongOnlyEnv, SeqLongOnlyEnvConfig
# ====================================================================
# Environment utils
# -----------------

def custom_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess OHLCV dataframe with engineered features for RL trading.
    Expected columns: ["open", "high", "low", "close", "volume"]
    """

    df = df.copy().reset_index(drop=False)

    df["features_close"] = df["close"]
    df["features_open"] = df["open"]
    df["features_high"] = df["high"]
    df["features_low"] = df["low"]
    df["features_volume"] = df["volume"]
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
        ),
    )
    return transformed_env, vecnorm


def make_environment(train_df, test_df, cfg, train_num_envs=1, eval_num_envs=1):
    """Make environments for training and evaluation."""
    maker = functools.partial(env_maker, train_df, cfg)
    parallel_env = ParallelEnv(
        train_num_envs,
        EnvCreator(maker),
        serial_for_single=True,
    )
    parallel_env.set_seed(cfg.env.seed)

    # Create train environment and get its VecNormV2 instance
    train_env, train_vecnorm = apply_env_transforms(parallel_env)

    # Create eval environment with its own VecNormV2
    # Note: Each VecNormV2 will maintain its own running statistics
    # This is acceptable as eval will normalize observations consistently during evaluation
    maker = functools.partial(env_maker, test_df, cfg)
    eval_base_env = ParallelEnv(
        eval_num_envs,
        EnvCreator(maker),
        serial_for_single=True,
    )

    # Create eval environment with separate VecNormV2 (will compute its own statistics)
    eval_env, eval_vecnorm = apply_env_transforms(eval_base_env)

    return train_env, eval_env


# ====================================================================
# Collector and replay buffer
# ---------------------------


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

def make_discrete_iql_model(cfg, env, device):
    """Make discrete IQL agent."""
    # Define Actor Network
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
        net = SimpleCNNEncoder(
            input_shape=(w, num_features),
            output_shape=(1, 14),
            hidden_channels=64,
            kernel_size=3,
            activation="relu",
            final_activation="relu",
            dropout=0.1,
        )
        encoders.append(SafeModule(net, in_keys=key, out_keys=[f"encoding_{t}_{w}"]))

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
        in_keys=[f"encoding_{t}_{w}" for t, w in zip(time_frames, window_sizes)] + ["encoding_account_state"],
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
        in_keys=[f"encoding_{t}_{w}" for t, w in zip(time_frames, window_sizes)] + ["encoding_account_state"],
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
        in_keys=[f"encoding_{t}_{w}" for t, w in zip(time_frames, window_sizes)] + ["encoding_account_state"],
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