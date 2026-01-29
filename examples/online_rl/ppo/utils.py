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

from torchtrade.envs import SequentialTradingEnv, SequentialTradingEnvConfig, SequentialTradingEnvSLTP, SequentialTradingEnvSLTPConfig
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

def env_maker(df, cfg, device="cpu", max_traj_length=1, random_start=False):
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
            random_start=random_start
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
            random_start=random_start
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


def make_environment(train_df, test_df, cfg, train_num_envs=1, eval_num_envs=1,
                     max_train_traj_length=1,
                     max_eval_traj_length=1):
    """Make environments for training and evaluation."""
    maker = functools.partial(env_maker, train_df, cfg, max_traj_length=max_train_traj_length, random_start=True)
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
    maker = functools.partial(env_maker, test_df, cfg, max_traj_length=max_eval_traj_length)
    eval_base_env = ParallelEnv(
        eval_num_envs,
        EnvCreator(maker),
        serial_for_single=True,
    )
    max_eval_steps = test_df.shape[0]
    eval_env = apply_env_transforms(eval_base_env, max_eval_steps)

    return train_env, eval_env



# ====================================================================
# Model utils
# --------------------------------------------------------------------


def make_discrete_ppo_model(cfg, env, device):
    """Make discrete PPO agent."""
    # Define Actor Network
    activation = "tanh"
    action_spec = env.action_spec
    market_data_keys = [k for k in list(env.observation_spec.keys()) if k.startswith("market_data")]
    assert "account_state" in list(env.observation_spec.keys()), "Account state key not in observation spec"
    account_state_key = "account_state"
    # Define Actor Network
    time_frames = cfg.env.time_frames
    window_sizes = cfg.env.window_sizes

    encoders = []
    num_features = env.observation_spec[market_data_keys[0]].shape[-1]

    # Build the encoder
    for key, t, w in zip(market_data_keys, time_frames, window_sizes):
    
        model = SimpleCNNEncoder(input_shape=(w, num_features),
                            output_shape=(1, 14), # if None, the output shape will be the same as the input shape otherwise you have to provide the output shape (out_seq, out_feat)
                            hidden_channels=64,
                            kernel_size=3,
                            activation=activation,
                            final_activation=activation,
                            dropout=0.1)
        encoders.append(SafeModule(
            module=model,
            in_keys=key,
            out_keys=[f"encoding_{t}_{w}"],
        ).to(device))


    # Get account state dimension from environment
    account_state_dim = env.observation_spec[account_state_key].shape[-1]

    account_encoder_model = SimpleMLPEncoder(
        input_shape=(1, account_state_dim),
        output_shape=(1, 14),
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

    # Define the actor
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

    # Define on head for the policy
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

    # Define another head for the value
    value_net = MLP(
        activation_class=torch.nn.ReLU, #ACTIVATIONS[activation], #
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


def make_collector(cfg, train_env, actor_model_explore, compile_mode, device="cpu"):
    """Make collector.

    Args:
        cfg: Configuration object
        train_env: Training environment
        actor_model_explore: Actor model for exploration
        compile_mode: Compilation mode for the policy
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
    )
    collector.set_seed(cfg.env.seed)
    return collector



def log_metrics(logger, metrics, step):
    for metric_name, metric_value in metrics.items():
        logger.log_scalar(metric_name, metric_value, step)