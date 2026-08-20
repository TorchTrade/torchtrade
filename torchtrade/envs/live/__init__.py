"""Live trading environments for TorchTrade."""

from torchtrade.envs.core.live import ObservationFailurePolicy

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
    MarginMode as BybitMarginMode,
    PositionMode as BybitPositionMode,
)

# Every venue's margin enum is qualified here. They share a NAME and not their VALUES --
# core/binance `ISOLATED`, bitget and bybit `isolated`, okx `cross` where the others say
# `crossed` -- and those are API wire strings, so an unqualified one silently sends the
# wrong case to whichever venue it did not come from (#289).
from torchtrade.envs.live.okx import (
    OKXObservationClass,
    OKXFuturesOrderClass,
    OKXFuturesTorchTradingEnv,
    OKXFuturesTradingEnvConfig,
    OKXFuturesSLTPTorchTradingEnv,
    OKXFuturesSLTPTradingEnvConfig,
    MarginMode as OKXMarginMode,
    PositionMode as OKXPositionMode,
)

# Polymarket — prediction markets via the CLOB
from torchtrade.envs.live.polymarket import (
    MarketScanner,
    MarketScannerConfig,
    PolymarketMarket,
    PolymarketOrderExecutor,
    PolymarketBetEnv,
    PolymarketBetEnvConfig,
)

__all__ = [
    "ObservationFailurePolicy",
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
    "BybitMarginMode",
    "BybitPositionMode",
    # OKX
    "OKXObservationClass",
    "OKXFuturesOrderClass",
    "OKXFuturesTorchTradingEnv",
    "OKXFuturesTradingEnvConfig",
    "OKXFuturesSLTPTorchTradingEnv",
    "OKXFuturesSLTPTradingEnvConfig",
    "OKXMarginMode",
    "OKXPositionMode",
    # Polymarket
    "MarketScanner",
    "MarketScannerConfig",
    "PolymarketMarket",
    "PolymarketOrderExecutor",
    "PolymarketBetEnv",
    "PolymarketBetEnvConfig",
]
