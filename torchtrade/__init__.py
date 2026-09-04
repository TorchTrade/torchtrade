"""TorchTrade: reinforcement learning environments for trading."""

# The single source. Hatchling reads it from here (`[tool.hatch.version]`), so
# pyproject.toml declares the version dynamic rather than repeating the number.
# tests/test_version_pin.py guards that, because it was three copies once.
__version__ = "0.1.0"

from torchtrade.envs.offline import (
    SequentialTradingEnv,
    SequentialTradingEnvConfig,
    SequentialTradingEnvSLTP,
    SequentialTradingEnvSLTPConfig,
    OneStepTradingEnv,
    OneStepTradingEnvConfig,
    MarginMode,
)
