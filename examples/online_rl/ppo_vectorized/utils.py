from __future__ import annotations

from torchrl.envs import (
    DoubleToFloat,
    ExplorationType,
    FlattenObservation,
    RewardSum,
    InitTracker,
    Compose,
    TransformedEnv,
    StepCounter,
)
from torchrl.collectors import SyncDataCollector
import torchrl
import torch
from torchrl.modules import (
    ActorValueOperator,
    MLP,
    ProbabilisticActor,
    ValueOperator,
    SafeModule,
    SafeSequential,
)
from torchtrade.models.simple_encoders import SimpleCNNEncoder, SimpleMLPEncoder
from torchtrade.models import BatchNormMLP

from torchtrade.envs.offline import VectorizedSequentialTradingEnv, VectorizedSequentialTradingEnvConfig
from torchtrade.envs import SequentialTradingEnv, SequentialTradingEnvConfig
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

def env_maker(df, cfg, device="cpu", max_traj_length=1, random_start=False, num_envs=None):
    """Create a vectorized environment."""
    if cfg.name == "VectorizedSequentialTradingEnv":
        # Use provided num_envs or fallback to cfg.train_envs/num_envs
        envs_count = num_envs if num_envs is not None else cfg.num_envs
        
        config = VectorizedSequentialTradingEnvConfig(
            num_envs=envs_count,
            symbol=cfg.symbol,
            time_frames=cfg.time_frames,
            window_sizes=cfg.window_sizes,
            execute_on=cfg.execute_on,
            initial_cash=cfg.initial_cash,
            slippage=cfg.slippage,
            transaction_fee=cfg.transaction_fee,
            bankrupt_threshold=cfg.bankrupt_threshold,
            seed=cfg.seed,
            leverage=cfg.leverage,
            action_levels=cfg.action_levels,
            max_traj_length=max_traj_length,
            random_start=random_start
        )
        return VectorizedSequentialTradingEnv(df, config, feature_preprocessing_fn=custom_preprocessing)
    else:
        raise ValueError(f"Unknown environment: {cfg.name}")

def apply_env_transforms(env, max_steps, flatten_market_obs=True):
    """Apply standard transforms to the environment.

    Args:
        env: Base environment
        max_steps: Maximum steps for StepCounter
        flatten_market_obs: For vectorized env, this must be True

    Returns:
        transformed_env: Environment with transforms applied

    Note: Normalization is handled in the preprocessing function using StandardScaler
          to avoid VecNormV2 device issues.
    """
    transforms = [
        InitTracker(),
        DoubleToFloat(),
        RewardSum(),
        StepCounter(max_steps=max_steps),
    ]

    # Vectorized env requires flattened observations
    if flatten_market_obs:
        obs_keys = list(env.observation_spec.keys())
        market_keys = [k for k in obs_keys if k.startswith("market_data")]
        for key in market_keys:
            transforms.append(FlattenObservation(in_keys=[key], first_dim=-2, last_dim=-1))

    transformed_env = TransformedEnv(env, Compose(*transforms))
    return transformed_env


def make_environment(train_df, test_df, cfg, train_num_envs=None, eval_num_envs=None,
                     max_train_traj_length=1,
                     max_eval_traj_length=1,
                     max_train_eval_traj_length=None,
                     flatten_market_obs=True):
    """Make environments for training and evaluation."""
    # For vectorized env, num_envs is set in the config, not passed separately
    train_maker = lambda: env_maker(train_df, cfg, max_traj_length=max_train_traj_length, random_start=True)
    train_env = train_maker()
    
    max_train_steps = train_df.shape[0]
    train_env = apply_env_transforms(train_env, max_train_steps, flatten_market_obs=flatten_market_obs)

    # Create eval environment using scalar SequentialTradingEnv (has render_history)
    eval_config = SequentialTradingEnvConfig(
        symbol=cfg.symbol,
        time_frames=cfg.time_frames,
        window_sizes=cfg.window_sizes,
        execute_on=cfg.execute_on,
        initial_cash=cfg.initial_cash if not isinstance(cfg.initial_cash, (tuple, list)) else cfg.initial_cash[0],
        slippage=cfg.slippage,
        transaction_fee=cfg.transaction_fee,
        bankrupt_threshold=cfg.bankrupt_threshold,
        seed=cfg.seed,
        leverage=cfg.leverage,
        action_levels=cfg.action_levels,
        max_traj_length=max_eval_traj_length,
        random_start=False,
    )
    eval_env = SequentialTradingEnv(test_df, eval_config, feature_preprocessing_fn=custom_preprocessing)

    max_eval_steps = test_df.shape[0]
    eval_env = apply_env_transforms(eval_env, max_eval_steps, flatten_market_obs=flatten_market_obs)

    # Create train-eval environment: scalar env on train data, fixed start for consistent comparison
    train_eval_config = SequentialTradingEnvConfig(
        symbol=cfg.symbol,
        time_frames=cfg.time_frames,
        window_sizes=cfg.window_sizes,
        execute_on=cfg.execute_on,
        initial_cash=cfg.initial_cash if not isinstance(cfg.initial_cash, (tuple, list)) else cfg.initial_cash[0],
        slippage=cfg.slippage,
        transaction_fee=cfg.transaction_fee,
        bankrupt_threshold=cfg.bankrupt_threshold,
        seed=cfg.seed,
        leverage=cfg.leverage,
        action_levels=cfg.action_levels,
        max_traj_length=max_train_eval_traj_length or max_train_traj_length,
        random_start=False,
    )
    train_eval_env = SequentialTradingEnv(train_df, train_eval_config, feature_preprocessing_fn=custom_preprocessing)
    train_eval_env = apply_env_transforms(train_eval_env, max_train_steps, flatten_market_obs=flatten_market_obs)

    return train_env, eval_env, train_eval_env



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
    time_frames = cfg.time_frames
    window_sizes = cfg.window_sizes

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
        activation_class=ACTIVATIONS[activation],
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


def make_batchnorm_ppo_model(cfg, env, device):
    """Make PPO agent using BatchNormMLP as the shared backbone.

    For vectorized env, observations are already flat from FlattenObservation transform.
    Concatenates all flat observations and feeds through BatchNormMLP, then splits 
    into policy and value heads.
    """
    activation = "tanh"
    action_spec = env.action_spec
    market_data_keys = [k for k in list(env.observation_spec.keys()) if k.startswith("market_data")]
    assert "account_state" in list(env.observation_spec.keys())
    account_state_key = "account_state"

    # Observations are already flat from FlattenObservation transform
    total_input = sum(env.observation_spec[k].shape[-1] for k in market_data_keys)
    total_input += env.observation_spec[account_state_key].shape[-1]

    hidden_size = getattr(cfg.model, "hidden_size", 128)
    dropout = getattr(cfg.model, "dropout", 0.1)
    num_layers = getattr(cfg.model, "num_layers", 4)

    backbone = BatchNormMLP(
        input_size=total_input,
        output_size=hidden_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
    )

    from tensordict.nn import TensorDictModule
    common_module = TensorDictModule(
        module=backbone,
        in_keys=market_data_keys + [account_state_key],
        out_keys=["common_features"],
    ).to(device)

    # Policy head
    action_out_features = action_spec.n
    policy_net = MLP(
        in_features=hidden_size,
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
        distribution_class=torch.distributions.Categorical,
        return_log_prob=True,
        default_interaction_type=ExplorationType.RANDOM,
    )

    # Value head
    value_net = MLP(
        activation_class=ACTIVATIONS[activation],
        in_features=hidden_size,
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
    network_type = getattr(cfg, "model", {})
    network_type = getattr(network_type, "network_type", "batchnorm_mlp") if hasattr(network_type, "network_type") else "batchnorm_mlp"

    if network_type == "batchnorm_mlp":
        common_module, policy_module, value_module = make_batchnorm_ppo_model(cfg, env, device=device)
    else:
        common_module, policy_module, value_module = make_discrete_ppo_model(cfg, env, device=device)

    # Wrap modules in a single ActorCritic operator
    actor_critic = ActorValueOperator(
        common_operator=common_module,
        policy_operator=policy_module,
        value_operator=value_module,
    ).to(device)

    with torch.no_grad():
        if network_type == "batchnorm_mlp":
            actor_critic.eval()
            td = env.fake_tensordict().to(actor_critic.device)
            actor_critic(td)
            del td
            actor_critic.train()
        else:
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