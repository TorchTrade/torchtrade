"""Coverage tracking transform for environments with random resets.

This module provides a TorchRL transform that tracks which starting positions
have been visited during training with random episode resets.
"""

from typing import Dict, Any, Optional
import numpy as np
import torch
from torchrl.envs.transforms import Transform
from tensordict import TensorDictBase


class CoverageTracker(Transform):
    """Transform that tracks reset coverage for environments with random starts.

    Only tracks coverage during training (when environment has random_start=True).
    Does not affect test/evaluation environments.

    The tracker monitors which starting positions (reset indices) have been used
    during training and provides statistics about coverage distribution. This is
    useful for:
    - Ensuring comprehensive coverage of training data
    - Identifying under-sampled regions of the dataset
    - Curriculum learning strategies
    - Detecting overfitting to specific market conditions

    Usage:
        from torchrl.collectors import SyncDataCollector
        from torchrl.envs import TransformedEnv, Compose, InitTracker, DoubleToFloat, RewardSum
        from torchtrade.envs.transforms import CoverageTracker

        # Create environment with standard transforms
        env = TransformedEnv(
            base_env,
            Compose(
                InitTracker(),
                DoubleToFloat(),
                RewardSum(),
            )
        )

        # Create coverage tracker for postproc
        coverage_tracker = CoverageTracker()

        # Use coverage tracker as postproc in collector
        collector = SyncDataCollector(
            env,
            policy,
            frames_per_batch=1000,
            total_frames=100000,
            device="cuda",
            postproc=coverage_tracker,  # Process batches after collection
        )

        # During training, access coverage stats:
        for batch in collector:
            # ... train on batch ...

            # Log coverage periodically
            stats = coverage_tracker.get_coverage_stats()
            if stats["enabled"]:
                print(f"Coverage: {stats['coverage']:.2%}")
                print(f"Visited: {stats['visited_positions']} / {stats['total_positions']}")

    Attributes:
        _coverage_counts: Array tracking visit count for each starting position
        _total_resets: Total number of resets performed
        _enabled: Whether tracking is enabled (auto-detected based on random_start)
    """

    def __init__(self):
        """Initialize the CoverageTracker transform.

        CoverageTracker should be used as a postproc in SyncDataCollector.
        It reads reset_index from collected batches and aggregates coverage statistics.
        """
        super().__init__()
        self._coverage_counts: Optional[np.ndarray] = None
        self._total_resets: int = 0
        self._enabled: bool = True
        self._num_positions: Optional[int] = None

    def _reset(
        self, tensordict: TensorDictBase, tensordict_reset: TensorDictBase
    ) -> TensorDictBase:
        """Called during environment reset - only initializes coverage tracking.

        Coverage tracking now happens in forward() by reading reset_index from batches.
        This eliminates IPC overhead and moves work outside the critical reset path.

        Args:
            tensordict: Input tensordict (unused)
            tensordict_reset: Output tensordict from environment reset

        Returns:
            tensordict_reset: Unchanged output tensordict
        """
        # Initialize coverage tracking on first reset
        if self._coverage_counts is None:
            base_env = self._get_base_env()

            # Check if this is a ParallelEnv
            from torchrl.envs import ParallelEnv
            parallel_env = None
            if isinstance(base_env, ParallelEnv):
                parallel_env = base_env
            elif hasattr(base_env, 'parallel_env'):
                parallel_env = base_env.parallel_env

            if parallel_env is not None:
                # For ParallelEnv, create a test instance to check configuration
                if hasattr(parallel_env, 'create_env_fn'):
                    try:
                        test_env = parallel_env.create_env_fn[0]()
                        if hasattr(test_env, 'sampler') and hasattr(test_env, 'random_start'):
                            if test_env.random_start:
                                sampler = test_env.sampler
                                self._num_positions = len(sampler._exec_times_arr)
                                self._coverage_counts = np.zeros(self._num_positions, dtype=np.int32)
                                self._enabled = True
                            else:
                                self._enabled = False
                        else:
                            self._enabled = False
                        test_env.close()
                    except Exception as e:
                        print(f"CoverageTracker: Failed to initialize from ParallelEnv: {e}")
                        self._enabled = False
                else:
                    self._enabled = False
            # Check if this environment should track coverage (non-parallel case)
            elif hasattr(base_env, 'sampler') and hasattr(base_env, 'random_start'):
                # Only track if env has random_start enabled
                if base_env.random_start:
                    sampler = base_env.sampler
                    self._num_positions = len(sampler._exec_times_arr)
                    self._coverage_counts = np.zeros(self._num_positions, dtype=np.int32)
                    self._enabled = True
                else:
                    # Disable tracking for sequential/test environments
                    self._enabled = False
            else:
                # Environment doesn't support coverage tracking
                self._enabled = False

        # Track coverage from reset_index (if available in tensordict)
        # This handles both single-env resets and collector postproc
        # Note: ParallelEnv doesn't propagate reset_index in direct reset() calls
        # For ParallelEnv, use CoverageTracker as collector postproc instead
        if self._enabled and self._coverage_counts is not None and "reset_index" in tensordict_reset.keys():
            reset_indices = tensordict_reset.get("reset_index")

            # Handle batched (ParallelEnv) vs single reset
            if isinstance(reset_indices, torch.Tensor):
                if reset_indices.ndim > 0:
                    # Batched: multiple workers (ParallelEnv)
                    for idx in reset_indices.flatten():
                        idx = int(idx.item())
                        if 0 <= idx < len(self._coverage_counts):
                            self._coverage_counts[idx] += 1
                            self._total_resets += 1
                else:
                    # Single env, scalar tensor
                    reset_idx = int(reset_indices.item())
                    if 0 <= reset_idx < len(self._coverage_counts):
                        self._coverage_counts[reset_idx] += 1
                        self._total_resets += 1
            else:
                # Non-tensor (shouldn't happen, but handle it)
                reset_idx = int(reset_indices)
                if 0 <= reset_idx < len(self._coverage_counts):
                    self._coverage_counts[reset_idx] += 1
                    self._total_resets += 1

        return tensordict_reset

    def _get_base_env(self):
        """Navigate through wrapped environments to find the base environment.

        TorchRL environments can be wrapped in multiple layers (TransformedEnv,
        ParallelEnv, etc.). This method unwraps them to find the actual base
        environment that has the sampler.

        Returns:
            The unwrapped base environment
        """
        env = self.parent

        # Unwrap through TorchRL wrappers
        while hasattr(env, '_env'):
            env = env._env

        # Unwrap through ParallelEnv to get to actual env
        if hasattr(env, 'base_env'):
            env = env.base_env

        return env

    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        """Process collected batch and aggregate coverage from reset indices.

        This method is called by SyncDataCollector's postproc after data collection.
        It reads reset_index from the batch and updates coverage statistics.

        Args:
            tensordict: Collected batch from environment

        Returns:
            tensordict: Unchanged batch (coverage tracking is side-effect only)
        """
        # Only process if tracking is enabled and we have reset indices in batch
        if self._enabled and self._coverage_counts is not None and "reset_index" in tensordict.keys():
            # Get all reset indices from the batch
            # For ParallelEnv, each worker adds its own reset_index to its timesteps
            reset_indices = tensordict.get("reset_index")

            # Handle both batched (ParallelEnv) and unbatched (single env) cases
            if reset_indices.ndim > 0:
                # Flatten to 1D if needed (batch dimension)
                reset_indices = reset_indices.flatten()

                # Use torch.unique to count occurrences efficiently
                unique_indices, counts = torch.unique(reset_indices, return_counts=True)

                # Update coverage counts (convert to numpy for indexing)
                for idx, count in zip(unique_indices.cpu().numpy(), counts.cpu().numpy()):
                    idx = int(idx)
                    if 0 <= idx < len(self._coverage_counts):
                        self._coverage_counts[idx] += int(count)
                        self._total_resets += int(count)
            else:
                # Single reset index (scalar)
                idx = int(reset_indices.item())
                if 0 <= idx < len(self._coverage_counts):
                    self._coverage_counts[idx] += 1
                    self._total_resets += 1

        return tensordict

    def get_coverage_stats(self) -> Dict[str, Any]:
        """Return coverage statistics.

        Returns:
            Dictionary containing coverage metrics:
            - enabled (bool): Whether tracking is active
            - total_positions (int): Total number of starting positions available
            - visited_positions (int): Number of unique positions used as resets
            - unvisited_positions (int): Number of positions never used
            - coverage (float): Fraction of positions visited, range [0, 1]
            - total_resets (int): Total number of resets performed
            - mean_visits_per_position (float): Average visits per position
            - max_visits (int): Maximum visits to any single position
            - min_visits (int): Minimum visits to any single position
            - std_visits (float): Standard deviation of visit distribution
            - coverage_entropy (float): Shannon entropy of visit distribution (higher = more uniform)

            If tracking is disabled, returns:
            - enabled (bool): False
            - message (str): Reason why tracking is disabled
        """
        if not self._enabled or self._coverage_counts is None:
            return {
                "enabled": False,
                "message": "Coverage tracking disabled (not a random-start training env)"
            }

        total_positions = len(self._coverage_counts)
        visited_positions = np.sum(self._coverage_counts > 0)
        coverage_fraction = visited_positions / total_positions if total_positions > 0 else 0.0

        return {
            "enabled": True,
            "total_positions": int(total_positions),
            "visited_positions": int(visited_positions),
            "unvisited_positions": int(total_positions - visited_positions),
            "coverage": float(coverage_fraction),
            "total_resets": int(self._total_resets),
            "mean_visits_per_position": float(self._coverage_counts.mean()),
            "max_visits": int(self._coverage_counts.max()),
            "min_visits": int(self._coverage_counts.min()),
            "std_visits": float(self._coverage_counts.std()),
            "coverage_entropy": float(self._compute_entropy()),
        }

    def _compute_entropy(self) -> float:
        """Compute Shannon entropy of coverage distribution.

        Entropy measures the uniformity of the visit distribution:
        - Higher entropy = more uniform coverage (good)
        - Lower entropy = concentrated on few positions (bad)
        - Max entropy = log(N) where N is number of positions

        Returns:
            Shannon entropy in nats (natural logarithm)
        """
        if self._coverage_counts is None or self._total_resets == 0:
            return 0.0

        # Normalize to probability distribution
        probs = self._coverage_counts / self._total_resets
        probs = probs[probs > 0]  # Filter zeros to avoid log(0)

        # Shannon entropy: H = -sum(p * log(p))
        entropy = -np.sum(probs * np.log(probs + 1e-10))
        return entropy

    def get_coverage_distribution(self) -> Optional[np.ndarray]:
        """Get the raw coverage counts array.

        Returns:
            Copy of coverage counts array, or None if tracking disabled.
            Shape: (num_positions,) with counts[i] = number of visits to position i
        """
        if self._coverage_counts is not None:
            return self._coverage_counts.copy()
        return None

    def reset_coverage(self):
        """Reset coverage statistics.

        This can be useful for curriculum learning where you want to track
        coverage separately for different training phases.
        """
        if self._coverage_counts is not None:
            self._coverage_counts.fill(0)
            self._total_resets = 0
