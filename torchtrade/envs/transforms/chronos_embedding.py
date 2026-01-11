"""Chronos embedding transform for time series feature extraction.

This module provides a TorchRL transform that uses pretrained Chronos models
to embed time series market data into fixed-size feature vectors.
"""

from typing import List, Optional, Union
import torch
import torch.nn as nn
from torchrl.envs.transforms import Transform
from torchrl.data import CompositeSpec, UnboundedContinuousTensorSpec
from tensordict import TensorDictBase
import warnings


class ChronosEmbeddingTransform(Transform):
    """Transform that embeds time series observations using pretrained Chronos models.

    Similar to VC1Transform for vision, but adapted for financial time series.
    Uses Amazon's Chronos T5-based forecasting models to extract meaningful
    embeddings from OHLCV market data.

    The transform processes market data observations (shape: [window_size, num_features])
    and produces fixed-size embedding vectors suitable for RL policy networks.

    Available Chronos models:
        - chronos-t5-tiny (8M params) - For testing and CI
        - chronos-t5-mini (20M params) - Small deployments
        - chronos-t5-small (46M params) - Balanced
        - chronos-t5-base (200M params) - Standard
        - chronos-t5-large (710M params) - Best performance (default)

    Usage:
        from torchrl.envs import TransformedEnv, Compose, InitTracker, RewardSum
        from torchtrade.envs.transforms import ChronosEmbeddingTransform

        # Create environment with Chronos embedding
        env = TransformedEnv(
            base_env,
            Compose(
                ChronosEmbeddingTransform(
                    in_keys=["market_data_1Minute_12"],
                    out_keys=["chronos_embedding"],
                    model_name="amazon/chronos-t5-large",
                    aggregation="mean"
                ),
                InitTracker(),
                RewardSum(),
            )
        )

    Args:
        in_keys: List of input keys to transform (market data observations)
        out_keys: List of output keys for embeddings
        model_name: Chronos model name (default: "amazon/chronos-t5-large")
        aggregation: How to aggregate feature embeddings - "mean", "max", or "concat"
        del_keys: Whether to delete input keys after transformation (default: True)
        device: Device for model inference (default: "cuda" if available else "cpu")
        torch_dtype: Torch dtype for model (default: torch.bfloat16)

    Attributes:
        pipeline: Chronos pipeline (loaded lazily on first use)
        _initialized: Whether the model has been loaded
        embedding_dim: Dimension of output embeddings
    """

    def __init__(
        self,
        in_keys: List[str],
        out_keys: List[str],
        model_name: str = "amazon/chronos-t5-large",
        aggregation: str = "mean",
        del_keys: bool = True,
        device: Optional[Union[str, torch.device]] = None,
        torch_dtype: torch.dtype = torch.bfloat16,
    ):
        """Initialize ChronosEmbeddingTransform with lazy model loading.

        Args:
            in_keys: Market data keys to embed (e.g., ["market_data_1Minute_12"])
            out_keys: Output embedding keys (e.g., ["chronos_embedding"])
            model_name: Chronos model identifier from HuggingFace
            aggregation: Aggregation method for multi-feature embeddings
            del_keys: Remove input keys after transformation
            device: Computation device (auto-detect if None)
            torch_dtype: Model precision (bfloat16 recommended for memory)
        """
        super().__init__(in_keys=in_keys, out_keys=out_keys)

        if len(in_keys) != len(out_keys):
            raise ValueError(
                f"in_keys and out_keys must have same length, got {len(in_keys)} vs {len(out_keys)}"
            )

        if aggregation not in ["mean", "max", "concat"]:
            raise ValueError(
                f"aggregation must be 'mean', 'max', or 'concat', got '{aggregation}'"
            )

        self.model_name = model_name
        self.aggregation = aggregation
        self.del_keys = del_keys
        self.torch_dtype = torch_dtype

        # Auto-detect device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Lazy initialization - model loaded on first forward pass
        self.pipeline = None
        self._initialized = False
        self.model = None

        # Embedding dimension (set after initialization based on model)
        self.embedding_dim = None

        # Cache for observation specs
        self._transformed_spec = None

    def _init(self) -> None:
        """Lazy initialization of Chronos model on first use.

        This follows the VC1Transform pattern - deferring model loading until
        first forward pass allows appending to environments without checking
        compatibility upfront.

        Raises:
            ImportError: If chronos-forecasting package is not installed
            Exception: If model loading fails
        """
        if self._initialized:
            return

        try:
            from chronos import ChronosPipeline
        except ImportError:
            raise ImportError(
                "chronos-forecasting package required for ChronosEmbeddingTransform. "
                "Install with: pip install git+https://github.com/amazon-science/chronos-forecasting.git"
            )

        try:
            # Load Chronos pipeline
            self.pipeline = ChronosPipeline.from_pretrained(
                self.model_name,
                device_map=str(self.device),
                torch_dtype=self.torch_dtype,
            )


            embedding_dim, _ = self.pipeline.embed(torch.zeros((1, 10)))
            self.embedding_dim = embedding_dim[:, -1, :].shape[-1]
            self._initialized = True

        except Exception as e:
            raise RuntimeError(
                f"Failed to load Chronos model '{self.model_name}': {e}\n"
                f"Make sure the model name is valid and you have internet connection "
                f"for first-time download."
            ) from e

    @torch.no_grad()
    def _extract_embedding(self, series: torch.Tensor) -> torch.Tensor:
        """Extract embedding from a single time series using Chronos encoder.

        Args:
            series: Time series tensor of shape (sequence_length,)

        Returns:
            Embedding tensor of shape (embedding_dim,)
        """
        embeddings, _ = self.pipeline.embed(series.cpu()) # chronos error when we keep it on GPU! BAD
        return embeddings[:, -1, :]

    def _aggregate_embeddings(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Aggregate feature embeddings using configured strategy.

        Args:
            embeddings: Stacked embeddings of shape (num_features, embedding_dim)

        Returns:
            Aggregated embedding of shape (embedding_dim,) or (num_features * embedding_dim,)
        """
        if self.aggregation == "mean":
            return embeddings.mean(dim=0)
        if self.aggregation == "max":
            return embeddings.max(dim=0)[0]
        return embeddings.flatten()  # concat

    def _apply_transform(self, obs: torch.Tensor) -> torch.Tensor:
        """Transform market data observation into embedding.

        Args:
            obs: Market data tensor of shape (window_size,) or (window_size, num_features)

        Returns:
            Embedding tensor of shape (embedding_dim,) or (num_features * embedding_dim,)
        """
        # Lazy initialize on first use
        if not self._initialized:
            self._init()

        # Convert 1D to 2D for uniform handling
        if obs.ndim == 1:
            obs = obs.unsqueeze(-1)  # (window_size,) -> (window_size, 1)

        if obs.ndim != 2:
            raise ValueError(f"Expected 1D or 2D tensor, got shape {obs.shape}")

        # Extract embeddings for all features
        window_size, num_features = obs.shape
        embeddings = torch.stack([
            self._extract_embedding(obs[:, i]) for i in range(num_features)
        ])  # (num_features, embedding_dim)

        return self._aggregate_embeddings(embeddings)

    def _call(self, tensordict: TensorDictBase) -> TensorDictBase:
        """Process tensordict and transform specified keys.

        This method handles batched observations from parallel environments,
        following the VC1Transform pattern.

        Args:
            tensordict: Input tensordict with market data observations

        Returns:
            Transformed tensordict with embedding keys added
        """
        # Lazy initialize if needed
        if not self._initialized:
            self._init()

        # Process each in_key -> out_key pair
        for in_key, out_key in zip(self.in_keys, self.out_keys):
            if in_key not in tensordict.keys():
                continue

            obs = tensordict.get(in_key)

            # Handle batched observations
            if obs.ndim > 2:
                # Batched: flatten all batch dimensions
                batch_shape = obs.shape[:-2]
                obs_flat = obs.flatten(end_dim=-3)  # Flatten to (batch_size, window, features)

                # Process batch with list comprehension
                embeddings = torch.stack([self._apply_transform(o) for o in obs_flat])
                embeddings = embeddings.view(*batch_shape, -1)
            else:
                # Unbatched: single observation
                embeddings = self._apply_transform(obs)

            # Add transformed output
            tensordict.set(out_key, embeddings)

            # Optionally delete input key
            if self.del_keys:
                del tensordict[in_key]

        return tensordict

    forward = _call  # Alias for compatibility

    def _reset(self, tensordict: TensorDictBase, tensordict_reset: TensorDictBase) -> TensorDictBase:                        
        """Apply transform to reset observations."""                                                                         
        return self._call(tensordict_reset) 

    def transform_observation_spec(self, observation_spec: CompositeSpec) -> CompositeSpec:
        """Update observation spec with embedding dimensions.

        Args:
            observation_spec: Original observation spec

        Returns:
            Updated spec with embedding keys
        """
        if self._transformed_spec is not None:
            return self._transformed_spec

        # Lazy initialize to get embedding_dim
        if not self._initialized:
            self._init()

        spec = observation_spec.clone()

        for in_key, out_key in zip(self.in_keys, self.out_keys):
            if in_key not in spec.keys():
                warnings.warn(
                    f"Key '{in_key}' not found in observation_spec, skipping",
                    UserWarning
                )
                continue

            # Get input shape to determine output dimension
            in_spec = spec[in_key]

            if hasattr(in_spec, 'shape') and len(in_spec.shape) >= 2:
                # Multi-feature: (window_size, num_features)
                num_features = in_spec.shape[-1]

                if self.aggregation == "concat":
                    out_dim = self.embedding_dim * num_features
                else:  # mean or max
                    out_dim = self.embedding_dim
            else:
                # Single feature or unknown shape
                out_dim = self.embedding_dim

            # Create output spec
            spec.set(
                out_key,
                UnboundedContinuousTensorSpec(
                    shape=(out_dim,),
                    device=self.device,
                    dtype=self.torch_dtype
                )
            )

            # Remove input spec if del_keys=True
            if self.del_keys:
                del spec[in_key]

        self._transformed_spec = spec
        return spec

    def to(self, dest: Union[torch.dtype, torch.device, str]) -> "ChronosEmbeddingTransform":
        """Move transform to specified device or dtype.

        Args:
            dest: Target device or dtype

        Returns:
            Self for method chaining
        """
        # Handle device changes
        if isinstance(dest, (torch.device, str)) and not isinstance(dest, torch.dtype):
            self.device = torch.device(dest) if isinstance(dest, str) else dest

            # Move model if already initialized
            if self._initialized and self._encoder is not None:
                self._encoder = self._encoder.to(self.device)

        # Handle dtype changes
        if isinstance(dest, torch.dtype):
            self.torch_dtype = dest

            # Convert model if already initialized
            if self._initialized and self._encoder is not None:
                self._encoder = self._encoder.to(dtype=dest)

        # Clear cached spec since device/dtype changed
        self._transformed_spec = None

        return super().to(dest)
