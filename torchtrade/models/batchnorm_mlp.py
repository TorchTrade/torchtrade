"""BatchNormMLP - MLP with BatchNorm and Dropout for trading."""

import torch
import torch.nn as nn


class BatchNormMLP(nn.Module):
    """MLP with BatchNorm and Dropout.

    Takes one or more input tensors, concatenates them, and processes through
    a deep MLP with batch normalization after each hidden layer.

    Args:
        input_size: Total input dimension.
        output_size: Output dimension.
        hidden_size: Hidden layer width.
        num_layers: Number of hidden layers.
        dropout: Dropout rate.
        activation: Activation class.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_size: int = 128,
        num_layers: int = 4,
        dropout: float = 0.2,
        activation: type = nn.LeakyReLU,
    ):
        super().__init__()

        layers = []
        in_features = input_size
        for _ in range(num_layers):
            layers.extend([
                nn.Linear(in_features, hidden_size),
                nn.BatchNorm1d(hidden_size),
                activation(),
                nn.Dropout(dropout),
            ])
            in_features = hidden_size

        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(hidden_size, output_size)

    def forward(self, *inputs: torch.Tensor) -> torch.Tensor:
        """Forward pass. Accepts one or more tensors that get concatenated along dim=-1.

        Flattens leading batch dims to 2D for BatchNorm1d, then restores shape.
        """
        x = torch.cat(inputs, dim=-1) if len(inputs) > 1 else inputs[0]
        leading = x.shape[:-1]
        x = x.reshape(-1, x.shape[-1])
        x = self.head(self.backbone(x))
        return x.reshape(*leading, x.shape[-1])
