"""Timestamp transform for recording execution times in TensorDicts."""

import time

from tensordict import NonTensorData, TensorDictBase
from torchrl.envs.transforms import Transform


class TimestampTransform(Transform):
    """Add Unix timestamps to each step/reset for dataset creation.

    Useful for:
    - Creating offline datasets from live trading runs
    - Debugging and analyzing live trading performance
    - Correlating trading decisions with real-world events

    Usage:
        from torchrl.envs import TransformedEnv
        from torchtrade.envs.transforms import TimestampTransform

        env = TransformedEnv(base_env, TimestampTransform())
        td = env.reset()  # td["timestamp"] contains reset time
        td = env.step(td.set("action", action))
        # td["next", "timestamp"] contains step time

    Args:
        out_key: Key to store the timestamp. Defaults to "timestamp".
    """

    def __init__(self, out_key: str = "timestamp"):
        # Empty in_keys/out_keys since we don't transform existing data
        super().__init__(in_keys=[], out_keys=[])
        self._out_key = out_key

    def _reset(
        self, tensordict: TensorDictBase, tensordict_reset: TensorDictBase
    ) -> TensorDictBase:
        """Add timestamp on environment reset."""
        tensordict_reset.set(self._out_key, NonTensorData(time.time()))
        return tensordict_reset

    def _step(
        self, tensordict: TensorDictBase, next_tensordict: TensorDictBase
    ) -> TensorDictBase:
        """Add timestamp to next tensordict on step."""
        next_tensordict.set(self._out_key, NonTensorData(time.time()))
        return next_tensordict
