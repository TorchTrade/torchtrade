from __future__ import annotations

import functools

import torch.nn
import torch.optim
from tensordict.nn import InteractionType, TensorDictModule
from tensordict.nn.distributions import NormalParamExtractor
from torch.distributions import Categorical
from torchrl.data import Categorical as CategoricalSpec
from torchrl.data import Bounded as BoundedSpec

from torchrl.collectors import SyncDataCollector
from torchrl.data import (
    Composite,
    LazyMemmapStorage,
    TensorDictPrioritizedReplayBuffer,
    TensorDictReplayBuffer,
)
from torchrl.data.datasets.d4rl import D4RLExperienceReplay
from torchrl.data.replay_buffers import SamplerWithoutReplacement
from torchrl.envs import (
    CatTensors,
    Compose,
    DMControlEnv,
    DoubleToFloat,
    EnvCreator,
    InitTracker,
    ParallelEnv,
    RewardSum,
    TransformedEnv,
)

from torchrl.envs.libs.gym import GymEnv, set_gym_backend
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.modules import (
    MLP,
    ProbabilisticActor,
    SafeModule,
    SafeSequential,
    TanhNormal,
    ValueOperator,
)
from torchrl.objectives import DiscreteIQLLoss, HardUpdate, IQLLoss, SoftUpdate
from torchrl.record import VideoRecorder
from torchrl.trainers.helpers.models import ACTIVATIONS
from trading_nets.architectures.tabl.tabl import BiNMTABLModel
import copy

# ====================================================================
# Environment utils
# -----------------


def env_maker(cfg, device="cpu", from_pixels=False):
    lib = cfg.env.backend
    if lib in ("gym", "gymnasium"):
        with set_gym_backend(lib):
            return GymEnv(
                cfg.env.name,
                device=device,
                from_pixels=from_pixels,
                pixels_only=False,
                categorical_action_encoding=True,
            )
    elif lib == "dm_control":
        env = DMControlEnv(
            cfg.env.name, cfg.env.task, from_pixels=from_pixels, pixels_only=False
        )
        return TransformedEnv(
            env, CatTensors(in_keys=env.observation_spec.keys(), out_key="observation")
        )
    else:
        raise NotImplementedError(f"Unknown lib {lib}.")


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


def make_environment(cfg, train_num_envs=1, eval_num_envs=1, logger=None):
    """Make environments for training and evaluation."""
    maker = functools.partial(env_maker, cfg)
    parallel_env = ParallelEnv(
        train_num_envs,
        EnvCreator(maker),
        serial_for_single=True,
    )
    parallel_env.set_seed(cfg.env.seed)

    train_env = apply_env_transforms(parallel_env)

    maker = functools.partial(env_maker, cfg, from_pixels=cfg.logger.video)
    eval_env = TransformedEnv(
        ParallelEnv(
            eval_num_envs,
            EnvCreator(maker),
            serial_for_single=True,
        ),
        train_env.transform.clone(),
    )
    if cfg.logger.video:
        eval_env.insert_transform(
            0, VideoRecorder(logger, tag="rendered", in_keys=["pixels"])
        )
    return train_env, eval_env


# ====================================================================
# Collector and replay buffer
# ---------------------------


def make_collector(cfg, train_env, actor_model_explore, compile_mode):
    """Make collector."""
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
    # data = D4RLExperienceReplay(
    #     dataset_id=rb_cfg.dataset,
    #     split_trajs=False,
    #     batch_size=rb_cfg.batch_size,
    #     # We use drop_last to avoid recompiles (and dynamic shapes)
    #     sampler=SamplerWithoutReplacement(drop_last=True),
    #     prefetch=4,
    #     direct_download=True,
    # )
    data = TensorDictReplayBuffer(
        pin_memory=False,
        prefetch=4,
        #split_trajs=False,
        storage=LazyMemmapStorage(rb_cfg.buffer_size),
        batch_size=rb_cfg.batch_size,
        #shared=shared,
        #sampler=SamplerWithoutReplacement(drop_last=True),
    )
    data.loads(rb_cfg.path)
    idx = data.storage._storage["index"].max().item()
    td = data.storage._storage[:idx]
    del data
    data = TensorDictReplayBuffer(
        pin_memory=False,
        prefetch=4,
        #split_trajs=False,
        storage=LazyMemmapStorage(idx),
        batch_size=rb_cfg.batch_size,
        #shared=shared,
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
#
# We give one version of the model for learning from pixels, and one for state.
# TorchRL comes in handy at this point, as the high-level interactions with
# these models is unchanged, regardless of the modality.
#



def make_discrete_iql_model(cfg, device):
    """Make discrete IQL agent."""
    # Define Actor Network
    action_spec = CategoricalSpec(3)
    # Define Actor Network
    import tensordict
    encodernet1min12 = BiNMTABLModel(input_shape=(1, 12, 14),
                        output_shape=(1, 14), # if None, the output shape will be the same as the input shape otherwise you have to provide the output shape (out_seq, out_feat)
                        hidden_seq_size=12,
                        hidden_feature_size=14,
                        num_heads=3,
                        activation="relu",
                        final_activation="relu",
                        dropout=0.1,
                        initializer="kaiming_uniform")
    encodernet5min8 = BiNMTABLModel(input_shape=(1, 8, 14),
                        output_shape=(1, 14), # if None, the output shape will be the same as the input shape otherwise you have to provide the output shape (out_seq, out_feat)
                        hidden_seq_size=8,
                        hidden_feature_size=14,
                        num_heads=3,
                        activation="relu",
                        final_activation="relu",
                        dropout=0.1,
                        initializer="kaiming_uniform")

    encodernet15min8 = BiNMTABLModel(input_shape=(1, 8, 14),
                        output_shape=(1, 14), # if None, the output shape will be the same as the input shape otherwise you have to provide the output shape (out_seq, out_feat)
                        hidden_seq_size=8,
                        hidden_feature_size=14,
                        num_heads=3,
                        activation="relu",
                        final_activation="relu",
                        dropout=0.1,
                        initializer="kaiming_uniform")

    encodernet1h24 = BiNMTABLModel(input_shape=(1, 24, 14),
                        output_shape=(1, 14), # if None, the output shape will be the same as the input shape otherwise you have to provide the output shape (out_seq, out_feat)
                        hidden_seq_size=24,
                        hidden_feature_size=14,
                        num_heads=3,
                        activation="relu",
                        final_activation="relu",
                        dropout=0.1,
                        initializer="kaiming_uniform")

    encoder1min12 = SafeModule(
        module=encodernet1min12,
        in_keys=["market_data_1Minute_12"],
        out_keys=["encoding1min"],
    ).to(device)
    encoder5min8 = SafeModule(
        module=encodernet5min8,
        in_keys=["market_data_5Minute_8"],
        out_keys=["encoding5min"],
    ).to(device)
    encoder15min8 = SafeModule(
        module=encodernet15min8,
        in_keys=["market_data_15Minute_8"],
        out_keys=["encoding15min"],
    ).to(device)
    encoder1h24 = SafeModule(
        module=encodernet1h24,
        in_keys=["market_data_1Hour_24"],
        out_keys=["encoding1h"],

    ).to(device)
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

    encoder = SafeSequential(encoder1min12, encoder5min8, encoder15min8, encoder1h24, account_state_encoder)

    actor_net = MLP(
        num_cells=cfg.model.hidden_sizes,
        out_features=3,
        activation_class=ACTIVATIONS[cfg.model.activation],
        device=device,
    )

    actor_module = SafeModule(
        module=actor_net,
        in_keys=["encoding1min", "encoding5min", "encoding15min", "encoding1h", "encoding_account_state"],
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
        out_features=3,
        activation_class=ACTIVATIONS[cfg.model.activation],
        device=device,
    )
    
    qvalue = SafeModule(
        module=qvalue_net,
        in_keys=["encoding1min", "encoding5min", "encoding15min", "encoding1h", "encoding_account_state"],
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
        in_keys=["encoding1min", "encoding5min", "encoding15min", "encoding1h", "encoding_account_state"],
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


def dump_video(module):
    if isinstance(module, VideoRecorder):
        module.dump()