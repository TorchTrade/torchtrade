"""Simple encoder architectures for TorchTrade examples.

These encoders provide basic implementations for processing sequential market data
without requiring external dependencies. They are designed to be drop-in replacements
for more sophisticated architectures while maintaining the same interface.
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional


class SimpleMLPEncoder(nn.Module):
    """
    Simple MLP encoder for sequential data.

    Flattens the input sequence and processes through MLP layers.
    Compatible with BiNMTABLModel interface from trading-nets.

    Args:
        input_shape: (sequence_length, num_features)
        output_shape: (out_seq_length, out_features) - if None, matches input
        hidden_sizes: List of hidden layer sizes
        activation: Activation function name
        dropout: Dropout rate
        final_activation: Final activation function name
    """

    def __init__(
        self,
        input_shape: Tuple[int, int],
        output_shape: Optional[Tuple[int, int]] = None,
        hidden_sizes: Tuple[int, ...] = (128, 128),
        activation: str = "relu",
        dropout: float = 0.1,
        final_activation: Optional[str] = "relu",
        **kwargs  # Accept extra args for compatibility
    ):
        super().__init__()

        seq_len, num_features = input_shape
        input_size = seq_len * num_features

        # Default output shape matches input
        if output_shape is None:
            output_shape = input_shape
        self.output_shape = output_shape
        out_seq_len, out_features = output_shape
        output_size = out_seq_len * out_features

        # Build MLP layers
        layers = []
        prev_size = input_size

        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(self._get_activation(activation))
            layers.append(nn.Dropout(dropout))
            prev_size = hidden_size

        # Output layer
        layers.append(nn.Linear(prev_size, output_size))
        if final_activation:
            layers.append(self._get_activation(final_activation))

        self.mlp = nn.Sequential(*layers)
        self.input_shape = input_shape

    def _get_activation(self, name: str) -> nn.Module:
        """Get activation function by name."""
        activations = {
            "relu": nn.ReLU(),
            "gelu": nn.GELU(),
            "tanh": nn.Tanh(),
            "sigmoid": nn.Sigmoid(),
            "elu": nn.ELU(),
        }
        return activations.get(name.lower(), nn.ReLU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, seq_len, features) or (*batch_dims, seq_len, features)

        Returns:
            Output tensor of shape (batch, out_seq_len, out_features) or (*batch_dims, out_seq_len, out_features)
        """
        # Handle extra batch dimensions (e.g., from parallel envs)
        original_shape = x.shape
        if x.ndim > 3:
            # Flatten all batch dimensions except last two (seq_len, features)
            x = x.reshape(-1, *x.shape[-2:])

        batch_size = x.shape[0]

        # Flatten sequence
        x_flat = x.reshape(batch_size, -1)

        # Process through MLP
        out = self.mlp(x_flat)

        # Reshape to output shape
        out = out.reshape(batch_size, *self.output_shape)

        # Restore original batch dimensions if needed
        if len(original_shape) > 3:
            out = out.reshape(*original_shape[:-2], *out.shape[-2:])

        # Squeeze sequence dimension if output is single timestep
        if self.output_shape[0] == 1:
            out = out.squeeze(-2)

        return out


class SimpleCNNEncoder(nn.Module):
    """
    Simple 1D CNN encoder for sequential data.

    Uses 1D convolutions to process temporal patterns in market data.
    Compatible with BiNMTABLModel interface from trading-nets.

    Args:
        input_shape: (sequence_length, num_features)
        output_shape: (out_seq_length, out_features) - if None, matches input
        hidden_channels: Number of channels in hidden layers
        kernel_size: Convolution kernel size
        activation: Activation function name
        dropout: Dropout rate
        final_activation: Final activation function name
    """

    def __init__(
        self,
        input_shape: Tuple[int, int],
        output_shape: Optional[Tuple[int, int]] = None,
        hidden_channels: int = 64,
        kernel_size: int = 3,
        activation: str = "relu",
        dropout: float = 0.1,
        final_activation: Optional[str] = "relu",
        **kwargs  # Accept extra args for compatibility
    ):
        super().__init__()

        seq_len, num_features = input_shape

        # Default output shape matches input
        if output_shape is None:
            output_shape = input_shape
        self.output_shape = output_shape
        out_seq_len, out_features = output_shape

        # Build CNN layers
        self.conv1 = nn.Conv1d(
            num_features, hidden_channels,
            kernel_size=kernel_size, padding=kernel_size//2
        )
        self.conv2 = nn.Conv1d(
            hidden_channels, hidden_channels,
            kernel_size=kernel_size, padding=kernel_size//2
        )
        self.conv3 = nn.Conv1d(
            hidden_channels, out_features,
            kernel_size=kernel_size, padding=kernel_size//2
        )

        self.activation = self._get_activation(activation)
        self.dropout = nn.Dropout(dropout)

        # Pooling/projection to match output sequence length
        if seq_len != out_seq_len:
            self.pool = nn.AdaptiveAvgPool1d(out_seq_len)
        else:
            self.pool = nn.Identity()

        self.final_activation = self._get_activation(final_activation) if final_activation else None
        self.input_shape = input_shape

    def _get_activation(self, name: str) -> nn.Module:
        """Get activation function by name."""
        activations = {
            "relu": nn.ReLU(),
            "gelu": nn.GELU(),
            "tanh": nn.Tanh(),
            "sigmoid": nn.Sigmoid(),
            "elu": nn.ELU(),
        }
        return activations.get(name.lower(), nn.ReLU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, seq_len, features) or (*batch_dims, seq_len, features)

        Returns:
            Output tensor of shape (batch, out_seq_len, out_features) or (*batch_dims, out_seq_len, out_features)
        """
        # Handle extra batch dimensions (e.g., from parallel envs)
        original_shape = x.shape
        if x.ndim > 3:
            # Flatten all batch dimensions except last two (seq_len, features)
            x = x.reshape(-1, *x.shape[-2:])

        # Transpose to (batch, features, seq_len) for Conv1d
        x = x.transpose(1, 2)

        # Apply convolutions
        x = self.conv1(x)
        x = self.activation(x)
        x = self.dropout(x)

        x = self.conv2(x)
        x = self.activation(x)
        x = self.dropout(x)

        x = self.conv3(x)

        # Pool to output sequence length
        x = self.pool(x)

        # Transpose back to (batch, seq_len, features)
        x = x.transpose(1, 2)

        if self.final_activation:
            x = self.final_activation(x)

        # Restore original batch dimensions if needed
        if len(original_shape) > 3:
            x = x.reshape(*original_shape[:-2], *x.shape[-2:])

        # Squeeze sequence dimension if output is single timestep
        if self.output_shape[0] == 1:
            x = x.squeeze(-2)

        return x


class SimpleTransformerEncoder(nn.Module):
    """
    Simple Transformer encoder for sequential data.

    Uses standard PyTorch TransformerEncoder for processing temporal patterns.
    Compatible with BiNMTABLModel interface from trading-nets.

    Args:
        input_shape: (sequence_length, num_features)
        output_shape: (out_seq_length, out_features) - if None, matches input
        hidden_feature_size: Hidden dimension size
        num_heads: Number of attention heads
        num_layers: Number of transformer layers
        activation: Activation function name
        dropout: Dropout rate
        final_activation: Final activation function name
    """

    def __init__(
        self,
        input_shape: Tuple[int, int],
        output_shape: Optional[Tuple[int, int]] = None,
        hidden_feature_size: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        activation: str = "relu",
        dropout: float = 0.1,
        final_activation: Optional[str] = "relu",
        **kwargs  # Accept extra args for compatibility
    ):
        super().__init__()

        seq_len, num_features = input_shape

        # Default output shape matches input
        if output_shape is None:
            output_shape = input_shape
        self.output_shape = output_shape
        out_seq_len, out_features = output_shape

        # Input projection
        self.input_proj = nn.Linear(num_features, hidden_feature_size)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_feature_size,
            nhead=num_heads,
            dim_feedforward=hidden_feature_size * 4,
            dropout=dropout,
            activation=activation,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output projection
        self.output_proj = nn.Linear(hidden_feature_size, out_features)

        # Pooling/projection to match output sequence length
        if seq_len != out_seq_len:
            self.pool = nn.AdaptiveAvgPool1d(out_seq_len)
        else:
            self.pool = nn.Identity()

        self.final_activation = self._get_activation(final_activation) if final_activation else None
        self.input_shape = input_shape

    def _get_activation(self, name: str) -> nn.Module:
        """Get activation function by name."""
        activations = {
            "relu": nn.ReLU(),
            "gelu": nn.GELU(),
            "tanh": nn.Tanh(),
            "sigmoid": nn.Sigmoid(),
            "elu": nn.ELU(),
        }
        return activations.get(name.lower(), nn.ReLU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, seq_len, features) or (*batch_dims, seq_len, features)

        Returns:
            Output tensor of shape (batch, out_seq_len, out_features) or (*batch_dims, out_seq_len, out_features)
        """
        # Handle extra batch dimensions (e.g., from parallel envs)
        original_shape = x.shape
        if x.ndim > 3:
            # Flatten all batch dimensions except last two (seq_len, features)
            x = x.reshape(-1, *x.shape[-2:])

        # Project input to hidden size
        x = self.input_proj(x)

        # Apply transformer
        x = self.transformer(x)

        # Project to output features
        x = self.output_proj(x)

        # Pool to output sequence length if needed
        if not isinstance(self.pool, nn.Identity):
            # Transpose for adaptive pooling
            x = x.transpose(1, 2)
            x = self.pool(x)
            x = x.transpose(1, 2)

        if self.final_activation:
            x = self.final_activation(x)

        # Restore original batch dimensions if needed
        if len(original_shape) > 3:
            x = x.reshape(*original_shape[:-2], *x.shape[-2:])

        # Squeeze sequence dimension if output is single timestep
        if self.output_shape[0] == 1:
            x = x.squeeze(-2)

        return x
