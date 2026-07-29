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
from sklearn.preprocessing import StandardScaler
from torchtrade.envs.offline import SequentialTradingEnv, SequentialTradingEnvConfig
# ====================================================================
# Environment utils
# -----------------

def custom_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess OHLCV dataframe with normalized features for RL trading.

    Expected columns: ["open", "high", "low", "close", "volume"]
    Index can be datetime or integer.

    Uses StandardScaler for normalization to avoid VecNormV2 device issues.
    """

    df = df.copy().reset_index(drop=False)

    df["features_open"] = df["open"]
    df["features_high"] = df["high"]
    df["features_low"] = df["low"]
    df["features_close"] = df["close"]
    df["features_volume"] = df["volume"]

    # Normalize features using StandardScaler
    scaler = StandardScaler()
    feature_cols = [col for col in df.columns if col.startswith("features_")]
    df[feature_cols] = scaler.fit_transform(df[feature_cols])

    df.fillna(0, inplace=True)

    return df


def env_maker(df, cfg, device="cpu"):
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
        # Spot: flat/long only. The default [-1, 0, 1] pairs a short level with
        # leverage=1, so the env clips it to flat and the policy carries a dead action.
        action_levels=[0, 1],
        seed=cfg.env.seed,
    )
    return SequentialTradingEnv(df, config, feature_preprocessing_fn=custom_preprocessing)



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


def make_offline_replay_buffer(rb_cfg, env):
    if rb_cfg.data_path == "synthetic":
        # Roll out the env so the keys match observation_spec; a hand-built td doesn't.
        td = env.rollout(rb_cfg.buffer_size, break_when_any_done=False).reshape(-1)
    elif "/" in rb_cfg.data_path and not rb_cfg.data_path.startswith((".", "/")):
        # An org/name repo id. Write ./relative or /absolute for an on-disk buffer --
        # hydra chdirs into its run dir, so probing the filesystem here is unreliable.
        from datasets import load_dataset
        from torchtrade.utils import dataset_to_td
        ds = load_dataset(rb_cfg.data_path, split="train")
        td = dataset_to_td(ds)
    else:
        td = tensordict.load(rb_cfg.data_path)

    # dataset_to_td only yields the columns the dataset actually has, and the loss needs
    # terminated. Without truncation info, done is the correct stand-in.
    if ("next", "terminated") not in td.keys(include_nested=True):
        td.set(("next", "terminated"), td.get(("next", "done")).clone())

    # Value estimators need these as (*batch, 1) to match state_value.
    for key in (("next", "reward"), ("next", "done"), ("next", "terminated")):
        value = td.get(key)
        if value.ndim == td.ndim:
            td.set(key, value.unsqueeze(-1))

    size = td.shape[0]
    data = TensorDictReplayBuffer(
        pin_memory=False,
        prefetch=4,
        storage=LazyMemmapStorage(size),
        batch_size=rb_cfg.batch_size,
        sampler=SamplerWithoutReplacement(drop_last=True),
    )
    data.extend(td)
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

    # From the env, not hardcoded: the head must track cfg.env.action_levels.
    num_actions = env.action_spec.n

    actor_net = MLP(
        num_cells=cfg.model.hidden_sizes,
        out_features=num_actions,
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
        out_features=num_actions,
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

    # A real observation rather than observation_spec.rand(): lazy init on actual data
    # keeps the first forward pass representative of what training will see.
    example_td = env.reset().to(device)
    with torch.no_grad(), set_exploration_type(ExplorationType.RANDOM):
        for net in model:
            net(example_td)

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