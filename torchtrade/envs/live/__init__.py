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

# OKX (use qualified imports to avoid shadowing Bybit's MarginMode/PositionMode)
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
