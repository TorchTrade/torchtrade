"""Live trading environments for TorchTrade."""

# Alpaca
from torchtrade.envs.live.alpaca import (
    AlpacaObservationClass,
    AlpacaOrderClass,
    AlpacaTorchTradingEnv,
    AlpacaTradingEnvConfig,
)

# Binance
from torchtrade.envs.live.binance import (
    BinanceObservationClass,
    BinanceFuturesOrderClass,
    BinanceFuturesTorchTradingEnv,
    BinanceFuturesTradingEnvConfig,
    TradeMode,
    MarginType,
    PositionSide,
)

# Bitget
from torchtrade.envs.live.bitget import (
    BitgetObservationClass,
    BitgetFuturesOrderClass,
    BitgetFuturesTorchTradingEnv,
    BitgetFuturesTradingEnvConfig,
)

# Bybit
from torchtrade.envs.live.bybit import (
    BybitObservationClass,
    BybitFuturesOrderClass,
    BybitFuturesTorchTradingEnv,
    BybitFuturesTradingEnvConfig,
    BybitFuturesSLTPTorchTradingEnv,
    BybitFuturesSLTPTradingEnvConfig,
    MarginMode,
    PositionMode,
)

# OKX
from torchtrade.envs.live.okx import (
    OKXObservationClass,
    OKXFuturesOrderClass,
    OKXFuturesTorchTradingEnv,
    OKXFuturesTradingEnvConfig,
    OKXFuturesSLTPTorchTradingEnv,
    OKXFuturesSLTPTradingEnvConfig,
)

__all__ = [
    # Alpaca
    "AlpacaObservationClass",
    "AlpacaOrderClass",
    "AlpacaTorchTradingEnv",
    "AlpacaTradingEnvConfig",
    # Binance
    "BinanceObservationClass",
    "BinanceFuturesOrderClass",
    "BinanceFuturesTorchTradingEnv",
    "BinanceFuturesTradingEnvConfig",
    "TradeMode",
    "MarginType",
    "PositionSide",
    # Bitget
    "BitgetObservationClass",
    "BitgetFuturesOrderClass",
    "BitgetFuturesTorchTradingEnv",
    "BitgetFuturesTradingEnvConfig",
    # Bybit
    "BybitObservationClass",
    "BybitFuturesOrderClass",
    "BybitFuturesTorchTradingEnv",
    "BybitFuturesTradingEnvConfig",
    "BybitFuturesSLTPTorchTradingEnv",
    "BybitFuturesSLTPTradingEnvConfig",
    "MarginMode",
    "PositionMode",
    # OKX
    "OKXObservationClass",
    "OKXFuturesOrderClass",
    "OKXFuturesTorchTradingEnv",
    "OKXFuturesTradingEnvConfig",
    "OKXFuturesSLTPTorchTradingEnv",
    "OKXFuturesSLTPTradingEnvConfig",
]
