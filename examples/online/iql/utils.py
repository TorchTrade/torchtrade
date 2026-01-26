from __future__ import annotations

import functools

import torch.nn
import torch.optim
from tensordict.nn import InteractionType
from torch.distributions import Categorical


from torchrl.collectors import SyncDataCollector
from torchrl.data import (
    Composite,
    LazyMemmapStorage,
    TensorDictPrioritizedReplayBuffer,
    TensorDictReplayBuffer,
)

from torchrl.envs import (
    Compose,
    DoubleToFloat,
    EnvCreator,
    InitTracker,
    ParallelEnv,
    RewardSum,
    TransformedEnv,
)

from torchtrade.envs.transforms import CoverageTracker

from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.modules import (
    MLP,
    ProbabilisticActor,
    SafeModule,
    SafeSequential,
)

from torchrl.objectives import DiscreteIQLLoss, HardUpdate
from torchrl.trainers.helpers.models import ACTIVATIONS
from torchtrade.models.simple_encoders import SimpleCNNEncoder
import copy
from torchtrade.envs.offline import SequentialTradingEnv, SequentialTradingEnvConfig
import pandas as pd

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
        )
        return SequentialTradingEnvSLTP(df, config, feature_preprocessing_fn=custom_preprocessing)
    elif cfg.env.name == "OneStepTradingEnv":
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
            include_hold_action=cfg.env.include_hold_action,
            quantity_per_trade=cfg.env.quantity_per_trade,
            trade_mode=cfg.env.trade_mode,
        )
        return OneStepTradingEnv(df, config, feature_preprocessing_fn=custom_preprocessing)
    else:
        raise ValueError(f"Unknown environment: {cfg.env.name}")



def apply_env_transforms(
    env,
):
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

    train_env = apply_env_transforms(parallel_env)

    maker = functools.partial(env_maker, test_df, cfg)
    eval_env = TransformedEnv(
        ParallelEnv(
            eval_num_envs,
            EnvCreator(maker),
            serial_for_single=True,
        ),
        train_env.transform.clone(),
    )

    # Create coverage tracker for postproc (used in collector)
    coverage_tracker = CoverageTracker()

    return train_env, eval_env, coverage_tracker


# ====================================================================
# Collector and replay buffer
# ---------------------------


def make_collector(cfg, train_env, actor_model_explore, compile_mode, postproc=None, device="cpu"):
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
        device="cpu",
        compile_policy={"mode": compile_mode} if compile_mode else False,
        cudagraph_policy={"warmup": 10} if cfg.compile.cudagraphs else False,
        postproc=postproc,  # Add coverage tracker as postproc
    )
    collector.set_seed(cfg.env.seed)
    return collector


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

# ====================================================================
# Model
# -----
#
# We give one version of the model for learning from pixels, and one for state.
# TorchRL comes in handy at this point, as the high-level interactions with
# these models is unchanged, regardless of the modality.
#



def make_discrete_iql_model(cfg, env, device):
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
    account_state_dim = env.observation_spec[account_state_key].shape[-1]

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
            in_features=account_state_dim,
            num_cells=[32],
            out_features=14,
            activation_class=ACTIVATIONS[cfg.model.activation],
            device=device,
        ),
        in_keys=[account_state_key],
        out_keys=["encoding_account_state"],
    ).to(device)

    encoder = SafeSequential(*encoders, account_state_encoder)

    # Define the actor
    actor_net = MLP(
        num_cells=cfg.model.hidden_sizes,
        out_features=action_spec.n,
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
        spec=Composite(action=action_spec).to(device),
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
        out_features=action_spec.n,
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

    # Define complete model
    model = torch.nn.ModuleList([actor, full_qvalue, full_value])


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

