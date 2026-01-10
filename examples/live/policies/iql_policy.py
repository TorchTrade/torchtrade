
from __future__ import annotations

from torch.distributions import Categorical
from torchrl.data import Categorical as CategoricalSpec
from tensordict.nn import InteractionType

from torchrl.data import (
    Composite)

from torchrl.modules import (
    MLP,
    ProbabilisticActor,
    SafeModule,
    SafeSequential,
)
from torchrl.envs.utils import ExplorationType, set_exploration_type
import torch

from torchrl.trainers.helpers.models import ACTIVATIONS
from torchtrade.models import SimpleCNNEncoder
import tensordict

def make_discrete_iql_model(device="cpu"):
    """Make discrete IQL agent."""
    # Define Actor Network
    action_spec = CategoricalSpec(3)
    # Define Actor Network
    encodernet1min12 = SimpleCNNEncoder(
        input_shape=(12, 14),
        output_shape=(1, 14),
        hidden_channels=64,
        kernel_size=3,
        activation="relu",
        final_activation="relu",
        dropout=0.1,
    )
    encodernet5min8 = SimpleCNNEncoder(
        input_shape=(8, 14),
        output_shape=(1, 14),
        hidden_channels=64,
        kernel_size=3,
        activation="relu",
        final_activation="relu",
        dropout=0.1,
    )

    encodernet15min8 = SimpleCNNEncoder(
        input_shape=(8, 14),
        output_shape=(1, 14),
        hidden_channels=64,
        kernel_size=3,
        activation="relu",
        final_activation="relu",
        dropout=0.1,
    )

    encodernet1h24 = SimpleCNNEncoder(
        input_shape=(24, 14),
        output_shape=(1, 14),
        hidden_channels=64,
        kernel_size=3,
        activation="relu",
        final_activation="relu",
        dropout=0.1,
    )

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
            activation_class=ACTIVATIONS["relu"],
            device=device,
        ),
        in_keys=["account_state"],
        out_keys=["encoding_account_state"],
    )

    encoder = SafeSequential(encoder1min12, encoder5min8, encoder15min8, encoder1h24, account_state_encoder)

    actor_net = MLP(
        num_cells=[256, 256],
        out_features=3,
        activation_class=ACTIVATIONS["relu"],
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
        actor(example_td)
    
    total_params = sum(p.numel() for p in actor.parameters())
    print(f"Total number of parameters: {total_params}")
    return actor