"""Bitget Futures live trading environments."""

from torchtrade.envs.live.bitget.observation import BitgetObservationClass
from torchtrade.envs.live.bitget.order_executor import (
    BitgetFuturesOrderClass,
    TradeMode,
    MarginMode,
    PositionMode,
    OrderStatus,
    PositionStatus,
)
from torchtrade.envs.live.bitget.env import (
    BitgetFuturesTorchTradingEnv,
    BitgetFuturesTradingEnvConfig,
)
from torchtrade.envs.live.bitget.env_sltp import (
    BitgetFuturesSLTPTorchTradingEnv,
    BitgetFuturesSLTPTradingEnvConfig,
)

__all__ = [
    "BitgetObservationClass",
    "BitgetFuturesOrderClass",
    "TradeMode",
    "MarginMode",
    "PositionMode",
    "OrderStatus",
    "PositionStatus",
    "BitgetFuturesTorchTradingEnv",
    "BitgetFuturesTradingEnvConfig",
    "BitgetFuturesSLTPTorchTradingEnv",
    "BitgetFuturesSLTPTradingEnvConfig",
]
