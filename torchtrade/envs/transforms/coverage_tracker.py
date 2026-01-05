"""Coverage tracking transform for environments with random resets.

This module provides a TorchRL transform that tracks which starting positions
have been visited during training with random episode resets.
"""

from typing import Dict, Any, Optional
import numpy as np
import torch
from torchrl.envs.transforms import Transform
from torchrl.envs import ParallelEnv
from tensordict import TensorDictBase

# Constants
_COVERAGE_ARRAY_BUFFER = 100  # Buffer size when growing coverage arrays dynamically


class CoverageTracker(Transform):
    """Transform that tracks both episode start diversity and full state visitation coverage.

    Only tracks coverage during training (when environment has random_start=True).
    Does not affect test/evaluation environments.

    The tracker monitors two types of coverage:
    1. Reset Coverage: Which starting positions (reset_index) are used for episode starts
    2. State Coverage: All timesteps (state_index) visited during episodes

    This dual tracking is useful for:
    - Ensuring comprehensive coverage of training data
    - Distinguishing between episode start diversity vs full trajectory coverage
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
                print(f"Reset Coverage: {stats['reset_coverage']:.2%}")
                print(f"State Coverage: {stats['state_coverage']:.2%}")

    Attributes:
        _reset_coverage_counts: Array tracking visit count for each starting position
        _state_coverage_counts: Array tracking visit count for all states
        _total_resets: Total number of episode resets
        _total_states: Total number of state visits
        _enabled: Whether tracking is enabled (auto-detected based on random_start)
    """

    def __init__(self):
        """Initialize the CoverageTracker transform.

        CoverageTracker should be used as a postproc in SyncDataCollector.
        It reads reset_index and state_index from collected batches and aggregates coverage statistics.
        """
        super().__init__()
        # Reset coverage tracking (episode start positions)
        self._reset_coverage_counts: Optional[np.ndarray] = None
        self._total_resets: int = 0

        # State coverage tracking (all timesteps visited)
        self._state_coverage_counts: Optional[np.ndarray] = None
        self._total_states: int = 0

        self._enabled: bool = True
        self._num_positions: Optional[int] = None

    def _reset(
        self, tensordict: TensorDictBase, tensordict_reset: TensorDictBase
    ) -> TensorDictBase:
        """Called during environment reset - initializes dual coverage tracking.

        Coverage tracking happens in forward() by reading reset_index and state_index from batches.
        This eliminates IPC overhead and moves work outside the critical reset path.

        Args:
            tensordict: Input tensordict (unused)
            tensordict_reset: Output tensordict from environment reset

        Returns:
            tensordict_reset: Unchanged output tensordict
        """
        # Initialize coverage tracking on first reset
        if self._reset_coverage_counts is None:
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
                                # Initialize both reset and state coverage tracking
                                self._reset_coverage_counts = np.zeros(self._num_positions, dtype=np.int64)
                                self._state_coverage_counts = np.zeros(self._num_positions, dtype=np.int64)
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
                    # Initialize both reset and state coverage tracking
                    self._reset_coverage_counts = np.zeros(self._num_positions, dtype=np.int64)
                    self._state_coverage_counts = np.zeros(self._num_positions, dtype=np.int64)
                    self._enabled = True
                else:
                    # Disable tracking for sequential/test environments
                    self._enabled = False
            else:
                # Environment doesn't support coverage tracking
                self._enabled = False

        # Track coverage from reset_index and state_index (if available in tensordict)
        # This handles both single-env resets and collector postproc
        # Note: ParallelEnv doesn't propagate these indices in direct reset() calls
        # For ParallelEnv, use CoverageTracker as collector postproc instead
        if self._enabled and self._reset_coverage_counts is not None:
            # Track reset_index
            if "reset_index" in tensordict_reset.keys():
                reset_indices = tensordict_reset.get("reset_index")

                # Handle batched (ParallelEnv) vs single reset
                if isinstance(reset_indices, torch.Tensor):
                    if reset_indices.ndim > 0:
                        # Batched: multiple workers (ParallelEnv)
                        for idx in reset_indices.flatten():
                            idx = int(idx.item())
                            if 0 <= idx < len(self._reset_coverage_counts):
                                self._reset_coverage_counts[idx] += 1
                                self._total_resets += 1
                    else:
                        # Single env, scalar tensor
                        reset_idx = int(reset_indices.item())
                        if 0 <= reset_idx < len(self._reset_coverage_counts):
                            self._reset_coverage_counts[reset_idx] += 1
                            self._total_resets += 1
                else:
                    # Non-tensor (shouldn't happen, but handle it)
                    reset_idx = int(reset_indices)
                    if 0 <= reset_idx < len(self._reset_coverage_counts):
                        self._reset_coverage_counts[reset_idx] += 1
                        self._total_resets += 1

            # Track state_index (same as reset_index during reset)
            if "state_index" in tensordict_reset.keys():
                state_indices = tensordict_reset.get("state_index")

                if isinstance(state_indices, torch.Tensor):
                    if state_indices.ndim > 0:
                        for idx in state_indices.flatten():
                            idx = int(idx.item())
                            if 0 <= idx < len(self._state_coverage_counts):
                                self._state_coverage_counts[idx] += 1
                                self._total_states += 1
                    else:
                        state_idx = int(state_indices.item())
                        if 0 <= state_idx < len(self._state_coverage_counts):
                            self._state_coverage_counts[state_idx] += 1
                            self._total_states += 1
                else:
                    state_idx = int(state_indices)
                    if 0 <= state_idx < len(self._state_coverage_counts):
                        self._state_coverage_counts[state_idx] += 1
                        self._total_states += 1

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
        """Process collected batch and aggregate coverage from reset and state indices.

        This method is called by SyncDataCollector's postproc after data collection.
        It reads both reset_index and state_index from the batch and updates coverage statistics.

        Args:
            tensordict: Collected batch from environment

        Returns:
            tensordict: Unchanged batch (coverage tracking is side-effect only)
        """
        # Auto-initialize on first forward() call if used as postproc (not in transform chain)
        # This happens when CoverageTracker.parent is None (not added to env transforms)
        if self._enabled and self._reset_coverage_counts is None and "reset_index" in tensordict.keys():
            reset_indices = tensordict.get("reset_index")
            if reset_indices.numel() > 0:
                # Initialize based on max index seen in first batch + buffer
                # This is a fallback for when _reset() doesn't get called (postproc usage)
                max_idx = int(reset_indices.max().item()) if reset_indices.ndim > 0 else int(reset_indices.item())
                # Use buffer to avoid frequent resizing
                initial_size = max_idx + _COVERAGE_ARRAY_BUFFER
                self._num_positions = initial_size
                self._reset_coverage_counts = np.zeros(initial_size, dtype=np.int64)
                self._state_coverage_counts = np.zeros(initial_size, dtype=np.int64)

                import warnings
                warnings.warn(
                    f"CoverageTracker auto-initialized with size {initial_size} from batch data. "
                    f"For best results, add CoverageTracker to the environment's transform chain so "
                    f"it can initialize from the environment's sampler.",
                    UserWarning,
                    stacklevel=2
                )

        # Coverage arrays must be initialized before tracking
        if not self._enabled or self._reset_coverage_counts is None or self._state_coverage_counts is None:
            return tensordict

        # Track reset coverage (episode start positions)
        if "reset_index" in tensordict.keys():
            reset_indices = tensordict.get("reset_index")

            # Handle both batched (ParallelEnv) and unbatched (single env) cases
            if reset_indices.ndim > 0:
                # Flatten to 1D if needed (batch dimension)
                reset_indices = reset_indices.flatten()

                # Use torch.unique to count occurrences efficiently
                unique_indices, counts = torch.unique(reset_indices, return_counts=True)

                # Update reset coverage counts (convert to numpy for indexing)
                for idx, count in zip(unique_indices.cpu().numpy(), counts.cpu().numpy()):
                    idx = int(idx)
                    if 0 <= idx < len(self._reset_coverage_counts):
                        self._reset_coverage_counts[idx] += int(count)
                        self._total_resets += int(count)
                    else:
                        # Index out of bounds - this indicates a problem with initialization
                        # or data corruption. Log warning but continue.
                        import warnings
                        warnings.warn(
                            f"CoverageTracker: reset_index {idx} out of bounds [0, {len(self._reset_coverage_counts)}). "
                            f"This may indicate incorrect initialization or data corruption.",
                            RuntimeWarning
                        )
            else:
                # Single reset index (scalar)
                idx = int(reset_indices.item())
                if 0 <= idx < len(self._reset_coverage_counts):
                    self._reset_coverage_counts[idx] += 1
                    self._total_resets += 1
                else:
                    import warnings
                    warnings.warn(
                        f"CoverageTracker: reset_index {idx} out of bounds [0, {len(self._reset_coverage_counts)}). "
                        f"This may indicate incorrect initialization or data corruption.",
                        RuntimeWarning
                    )

        # Track state coverage (all timesteps visited during episodes)
        if "state_index" in tensordict.keys():
            state_indices = tensordict.get("state_index")

            # Handle both batched (ParallelEnv) and unbatched (single env) cases
            if state_indices.ndim > 0:
                # Flatten to 1D if needed (batch dimension)
                state_indices = state_indices.flatten()

                # Use torch.unique to count occurrences efficiently
                unique_indices, counts = torch.unique(state_indices, return_counts=True)

                # Update state coverage counts (convert to numpy for indexing)
                for idx, count in zip(unique_indices.cpu().numpy(), counts.cpu().numpy()):
                    idx = int(idx)
                    if 0 <= idx < len(self._state_coverage_counts):
                        self._state_coverage_counts[idx] += int(count)
                        self._total_states += int(count)
                    else:
                        # Index out of bounds - this indicates a problem with initialization
                        # or data corruption. Log warning but continue.
                        import warnings
                        warnings.warn(
                            f"CoverageTracker: state_index {idx} out of bounds [0, {len(self._state_coverage_counts)}). "
                            f"This may indicate incorrect initialization or data corruption.",
                            RuntimeWarning
                        )
            else:
                # Single state index (scalar)
                idx = int(state_indices.item())
                if 0 <= idx < len(self._state_coverage_counts):
                    self._state_coverage_counts[idx] += 1
                    self._total_states += 1
                else:
                    import warnings
                    warnings.warn(
                        f"CoverageTracker: state_index {idx} out of bounds [0, {len(self._state_coverage_counts)}). "
                        f"This may indicate incorrect initialization or data corruption.",
                        RuntimeWarning
                    )

        return tensordict

    def get_coverage_stats(self) -> Dict[str, Any]:
        """Return dual coverage statistics for both reset and state tracking.

        INTERPRETING THE METRICS:

        Coverage Fraction (reset_coverage, state_coverage):
            - Range: [0.0, 1.0] where 0.0 = no coverage, 1.0 = complete coverage
            - Measures: What fraction of dataset positions have been visited
            - Good values: >0.8 indicates broad exploration
            - Bad values: <0.3 suggests concentrated sampling (potential overfitting)
            - Note: Reset coverage can be much lower than state coverage since episodes
              start at specific positions but then traverse many subsequent states

        Entropy (reset_entropy, state_entropy):
            - Range: [0, log(N)] where N = total_positions
            - Measures: Uniformity of the visit distribution
            - Maximum entropy = log(N): Perfectly uniform (all positions visited equally)
            - Low entropy: Concentrated on few positions (high variance in visit counts)
            - High entropy: Spread across many positions (uniform sampling)
            - Interpretation:
                * If coverage is high but entropy is low: Visiting many positions but
                  some are visited much more frequently than others
                * If coverage is low but entropy is high: Only visiting a subset, but
                  those positions are visited uniformly
                * Ideal: Both high coverage (>0.8) and high entropy (close to log(N))

        Reset vs State Coverage:
            - reset_coverage: Diversity of episode starting points
            - state_coverage: Diversity of all states seen during episodes (includes
              all timesteps from reset to done, not just episode starts)
            - Key insight: state_coverage should be significantly higher than
              reset_coverage because episodes traverse many states after starting.
              If they're similar, episodes might be too short or agent is resetting
              too frequently.

        Example interpretations:
            - reset_coverage=0.3, state_coverage=0.9: Good! Starting from 30% of
              positions but exploring 90% of dataset through episode trajectories
            - reset_coverage=0.5, state_coverage=0.5: Warning! Agent only seeing
              states near episode starts, not exploring forward in time
            - reset_entropy=3.2, max_entropy=log(1000)≈6.9: Moderate uniformity,
              some positions sampled more than others (check std_visits)

        Returns:
            Dictionary containing coverage metrics:
            - enabled (bool): Whether tracking is active
            - total_positions (int): Total number of positions in dataset

            Reset coverage (episode start diversity):
            - reset_visited (int): Number of unique starting positions used
            - reset_coverage (float): Fraction of positions used as episode starts [0, 1]
            - total_resets (int): Total number of episode resets
            - reset_mean_visits (float): Average visits per reset position
            - reset_max_visits (int): Maximum visits to any reset position
            - reset_min_visits (int): Minimum visits to any reset position
            - reset_std_visits (float): Standard deviation of reset visit distribution
            - reset_entropy (float): Shannon entropy of reset distribution (higher = more uniform)

            State coverage (full trajectory coverage):
            - state_visited (int): Number of unique states visited during episodes
            - state_coverage (float): Fraction of all states visited [0, 1]
            - total_states (int): Total number of state visits
            - state_mean_visits (float): Average visits per state
            - state_max_visits (int): Maximum visits to any state
            - state_min_visits (int): Minimum visits to any state
            - state_std_visits (float): Standard deviation of state visit distribution
            - state_entropy (float): Shannon entropy of state distribution (higher = more uniform)

            If tracking is disabled, returns:
            - enabled (bool): False
            - message (str): Reason why tracking is disabled
        """
        if not self._enabled or self._reset_coverage_counts is None or self._state_coverage_counts is None:
            return {
                "enabled": False,
                "message": "Coverage tracking disabled (not a random-start training env)"
            }

        total_positions = len(self._reset_coverage_counts)

        # Reset coverage statistics (episode start diversity)
        reset_visited = np.sum(self._reset_coverage_counts > 0)
        reset_coverage = reset_visited / total_positions if total_positions > 0 else 0.0

        # State coverage statistics (full trajectory coverage)
        state_visited = np.sum(self._state_coverage_counts > 0)
        state_coverage = state_visited / total_positions if total_positions > 0 else 0.0

        return {
            "enabled": True,
            "total_positions": int(total_positions),

            # Reset coverage (episode starts)
            "reset_visited": int(reset_visited),
            "reset_coverage": float(reset_coverage),
            "total_resets": int(self._total_resets),
            "reset_mean_visits": float(self._reset_coverage_counts.mean()),
            "reset_max_visits": int(self._reset_coverage_counts.max()),
            "reset_min_visits": int(self._reset_coverage_counts.min()),
            "reset_std_visits": float(self._reset_coverage_counts.std()),
            "reset_entropy": float(self._compute_entropy(self._reset_coverage_counts, self._total_resets)),

            # State coverage (all timesteps)
            "state_visited": int(state_visited),
            "state_coverage": float(state_coverage),
            "total_states": int(self._total_states),
            "state_mean_visits": float(self._state_coverage_counts.mean()),
            "state_max_visits": int(self._state_coverage_counts.max()),
            "state_min_visits": int(self._state_coverage_counts.min()),
            "state_std_visits": float(self._state_coverage_counts.std()),
            "state_entropy": float(self._compute_entropy(self._state_coverage_counts, self._total_states)),
        }

    def _compute_entropy(self, coverage_counts: np.ndarray, total_count: int) -> float:
        """Compute Shannon entropy of coverage distribution.

        Shannon entropy H measures the randomness/uniformity of a probability distribution.
        For coverage tracking, it tells us how uniformly we're sampling positions:

        Formula: H = -sum(p_i * log(p_i)) where p_i = count_i / total_count

        Interpretation:
            - H = 0: All visits to single position (minimum randomness)
            - H = log(N): Perfectly uniform across N positions (maximum randomness)
            - H in between: Some positions visited more than others

        Practical thresholds (for N=1000 positions, max_entropy ≈ 6.9):
            - H < 2.0: Very concentrated (bad - potential overfitting)
            - H = 2.0-4.0: Moderately concentrated
            - H = 4.0-6.0: Reasonably uniform
            - H > 6.0: Highly uniform (good - broad exploration)

        Why entropy matters:
            - High coverage + low entropy = Visiting many positions, but some way more
              than others (skewed sampling distribution)
            - Low coverage + high entropy = Only visiting subset of data, but uniformly
              (limited but balanced exploration)
            - High coverage + high entropy = Visiting most positions uniformly (ideal!)

        Args:
            coverage_counts: Array of visit counts per position
            total_count: Total number of visits

        Returns:
            Shannon entropy in nats (natural logarithm, not bits)
        """
        if coverage_counts is None or total_count == 0:
            return 0.0

        # Normalize to probability distribution
        probs = coverage_counts / total_count
        probs = probs[probs > 0]  # Filter zeros to avoid log(0)

        # Shannon entropy: H = -sum(p * log(p))
        entropy = -np.sum(probs * np.log(probs + 1e-10))
        return entropy

    def get_coverage_distribution(self) -> Dict[str, Optional[np.ndarray]]:
        """Get the raw coverage counts arrays for both reset and state tracking.

        Returns:
            Dictionary with:
            - reset_counts: Copy of reset coverage array, or None if tracking disabled
            - state_counts: Copy of state coverage array, or None if tracking disabled
            Shape: (num_positions,) with counts[i] = number of visits to position i
        """
        return {
            "reset_counts": self._reset_coverage_counts.copy() if self._reset_coverage_counts is not None else None,
            "state_counts": self._state_coverage_counts.copy() if self._state_coverage_counts is not None else None,
        }

    def reset_coverage(self):
        """Reset all coverage statistics.

        This can be useful for curriculum learning where you want to track
        coverage separately for different training phases.
        """
        if self._reset_coverage_counts is not None:
            self._reset_coverage_counts.fill(0)
            self._total_resets = 0
        if self._state_coverage_counts is not None:
            self._state_coverage_counts.fill(0)
            self._total_states = 0
