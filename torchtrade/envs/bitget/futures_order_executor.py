import logging
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Union

import ccxt

from torchtrade.envs.live.bitget.utils import normalize_symbol
from torchtrade.envs.core.common import TradeMode

logger = logging.getLogger(__name__)

# Bitget error codes that indicate no position exists (expected, not real errors)
BITGET_NO_POSITION_ERRORS = ["22002", "40773", "No position to close"]


class PositionMode(Enum):
    """
    Position mode for Bitget Futures.

    Bitget Futures supports two position modes:
    - One-way mode: Single net position per symbol.
    - Hedge mode: Separate long/short positions simultaneously.
    """
    ONE_WAY = "one_way_mode"      # Single net position (recommended)
    HEDGE = "hedge_mode"          # Separate long/short (advanced)


class MarginMode(Enum):
    """
    Margin mode for Bitget Futures positions.

    - ISOLATED: Margin is isolated per position. Losses are limited to
      that position's margin. Lower risk but requires more capital.
    - CROSSED: Margin is shared across all positions. Entire account
      balance can be used to prevent liquidation. Higher risk.
    """
    ISOLATED = "isolated"
    CROSSED = "crossed"

    def to_ccxt(self) -> str:
        """Convert to CCXT margin mode string.

        Returns:
            'isolated' for ISOLATED, 'cross' for CROSSED
        """
        return 'isolated' if self == MarginMode.ISOLATED else 'cross'


@dataclass
class OrderStatus:
    is_open: bool
    order_id: Optional[str]
    filled_qty: Optional[float]
    filled_avg_price: Optional[float]
    status: str
    side: str
    order_type: str


@dataclass
class PositionStatus:
    qty: float  # Positive for long, negative for short
    notional_value: float
    entry_price: float
    unrealized_pnl: float
    unrealized_pnl_pct: float
    mark_price: float
    leverage: int
    margin_mode: str
    liquidation_price: float


class BitgetFuturesOrderClass:
    """
    Order executor for Bitget Futures trading.

    Supports:
    - Long and short positions
    - Configurable leverage (1x-125x)
    - Market, limit, and stop orders
    - Bracket orders with stop-loss and take-profit
    - Demo (testnet) and production modes
    """

    def __init__(
        self,
        symbol: str,
        product_type: str = "USDT-FUTURES",  # V2 API: USDT-FUTURES, COIN-FUTURES, USDC-FUTURES
        trade_mode: TradeMode = TradeMode.QUANTITY,
        api_key: str = "",
        api_secret: str = "",
        passphrase: str = "",
        demo: bool = True,
        leverage: int = 1,
        margin_mode: MarginMode = MarginMode.ISOLATED,
        position_mode: PositionMode = PositionMode.ONE_WAY,
        client: Optional[object] = None,
    ):
        """
        Initialize the BitgetFuturesOrderClass.

        Args:
            symbol: The trading symbol (e.g., "BTC/USDT:USDT")
            product_type: Product type for V2 API (USDT-FUTURES, COIN-FUTURES, USDC-FUTURES)
            trade_mode: TradeMode.QUANTITY for unit-based orders
            api_key: Bitget API key
            api_secret: Bitget API secret
            passphrase: Bitget API passphrase (required!)
            demo: Whether to use demo/testnet (default: True for safety)
            leverage: Leverage to use (1-125, default: 1)
            margin_mode: ISOLATED (margin per position) or CROSSED (shared margin)
            position_mode: ONE_WAY (single net position) or HEDGE (separate long/short)
            client: Optional pre-configured CCXT exchange instance for dependency injection
        """
        import os
        import warnings

        # Check for deprecated environment variable names
        if os.getenv("BITGET_API_KEY"):
            warnings.warn(
                "Environment variable 'BITGET_API_KEY' is deprecated. "
                "Please use 'BITGETACCESSAPIKEY' instead.",
                DeprecationWarning,
                stacklevel=2
            )
        if os.getenv("BITGET_SECRET"):
            warnings.warn(
                "Environment variable 'BITGET_SECRET' is deprecated. "
                "Please use 'BITGETSECRETKEY' instead.",
                DeprecationWarning,
                stacklevel=2
            )
        if os.getenv("BITGET_PASSPHRASE"):
            warnings.warn(
                "Environment variable 'BITGET_PASSPHRASE' is deprecated. "
                "Please use 'BITGETPASSPHRASE' instead.",
                DeprecationWarning,
                stacklevel=2
            )

        self.symbol = normalize_symbol(symbol)
        # V2 API uses standardized product types
        self.product_type = product_type
        self.margin_coin = "USDT"  # Margin coin for USDT futures
        self.trade_mode = trade_mode
        self.demo = demo
        self.leverage = leverage
        self.margin_mode = margin_mode
        self.position_mode = position_mode
        self.last_order_id = None

        # Initialize CCXT client
        if client is not None:
            self.client = client
        else:
            try:
                self.client = ccxt.bitget({
                    'apiKey': api_key,
                    'secret': api_secret,
                    'password': passphrase,  # Bitget requires passphrase
                    'options': {
                        'defaultType': 'swap',  # Use futures/swap
                        'sandboxMode': demo,  # Enable testnet mode
                    }
                })

                # Enable sandbox/testnet mode
                if demo:
                    self.client.set_sandbox_mode(True)

            except Exception as e:
                raise ImportError(f"CCXT is required. Install with: pip install ccxt. Error: {e}")

        # Setup futures account
        self._setup_futures_account()

    def _calculate_unrealized_pnl_pct(self, qty: float, entry_price: float, mark_price: float) -> float:
        """Calculate unrealized PnL percentage.

        Args:
            qty: Position quantity (positive for long, negative for short)
            entry_price: Entry price
            mark_price: Current mark price

        Returns:
            Unrealized PnL percentage
        """
        if entry_price <= 0:
            return 0.0

        if qty > 0:
            return (mark_price - entry_price) / entry_price
        else:
            return (entry_price - mark_price) / entry_price

    def _get_opposite_side(self, side: str) -> str:
        """Get the opposite trading side.

        Args:
            side: Original side ("buy" or "sell")

        Returns:
            Opposite side ("sell" for "buy", "buy" for "sell")
        """
        return "sell" if side == "buy" else "buy"

    def _validate_and_prepare_stop_order(self, order_type: str, stop_price: Optional[float], params: dict) -> str:
        """Validate and prepare stop order parameters.

        Args:
            order_type: The order type (e.g., "stop", "stop_market")
            stop_price: The stop price trigger
            params: Order parameters dictionary to update

        Returns:
            Normalized order type for CCXT ("market" or "limit")

        Raises:
            ValueError: If stop_price is missing for stop orders
        """
        if stop_price is None:
            raise ValueError("stop_price is required for stop orders")
        params['stopPrice'] = stop_price
        return 'market' if order_type == 'stop_market' else 'limit'

    def _is_no_position_error(self, error: Exception) -> bool:
        """Check if error indicates no position exists.

        Args:
            error: The exception to check

        Returns:
            True if error indicates no position, False otherwise
        """
        error_msg = str(error)
        return any(err_code in error_msg for err_code in BITGET_NO_POSITION_ERRORS)

    def _build_order_params(self, reduce_only: bool = False, time_in_force: str = "GTC") -> dict:
        """Build common order parameters for Bitget orders.

        Args:
            reduce_only: If True, order will only reduce position
            time_in_force: Time in force setting

        Returns:
            Dictionary of order parameters
        """
        params = {
            'marginMode': self.margin_mode.value,
        }

        if time_in_force != "GTC":
            params['timeInForce'] = time_in_force

        if reduce_only:
            params['reduceOnly'] = True

        # Handle position mode
        if self.position_mode == PositionMode.HEDGE:
            params['tradeSide'] = 'close' if reduce_only else 'open'

        return params

    def _setup_futures_account(self):
        """Configure futures account settings."""
        try:
            params = {
                'productType': self.product_type,
                'marginCoin': self.margin_coin,
            }

            # Set position mode (one-way or hedge)
            try:
                position_mode_value = self.position_mode.value
                logger.info(f"Setting position mode to: {position_mode_value}")
                # Bitget API call to set position mode
                # Note: This is exchange-specific and may not be in CCXT unified API
                response = self.client.set_position_mode(
                    hedged=(self.position_mode == PositionMode.HEDGE),
                    symbol=self.symbol,
                    params=params
                )
                logger.info(f"Position mode set successfully to {position_mode_value}")
            except Exception as e:
                # May fail if already set or not supported
                logger.warning(f"Could not set position mode (may already be configured): {e}")

            # Set leverage using CCXT unified API
            self.client.set_leverage(
                leverage=self.leverage,
                symbol=self.symbol,
                params=params
            )

            # Set margin mode using CCXT
            try:
                margin_mode_ccxt = self.margin_mode.to_ccxt()
                logger.info(f"Setting margin mode to: {margin_mode_ccxt}")
                self.client.set_margin_mode(
                    marginMode=margin_mode_ccxt,
                    symbol=self.symbol
                )
                logger.info(f"Margin mode set successfully to {margin_mode_ccxt}")
            except Exception as e:
                logger.error(f"Error setting margin mode to {margin_mode_ccxt}: {e}")

        except Exception as e:
            # May fail if settings already configured - this is expected
            logger.warning(f"Could not setup futures account (may already be configured): {e}")

    def trade(
        self,
        side: str,
        quantity: float,
        order_type: str = "market",
        position_side: Optional[str] = None,
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None,
        take_profit: Optional[float] = None,
        stop_loss: Optional[float] = None,
        reduce_only: bool = False,
        time_in_force: str = "GTC",
    ) -> bool:
        """
        Execute a futures trade using CCXT unified API.

        Args:
            side: "buy" or "sell"
            quantity: Amount to trade in base asset units
            order_type: "market", "limit", "stop"
            position_side: Optional position side for hedge mode
            limit_price: Required for limit orders
            stop_price: Required for stop orders
            take_profit: Take profit price (creates separate TP order)
            stop_loss: Stop loss price (creates separate SL order)
            reduce_only: If True, only reduce position (no new positions)
            time_in_force: Time in force ("GTC", "IOC", "FOK")

        Returns:
            bool: True if order was submitted successfully
        """
        try:
            side = side.lower()
            order_type_lower = order_type.lower()

            # Build order parameters
            params = self._build_order_params(reduce_only=reduce_only, time_in_force=time_in_force)

            # Handle stop orders
            price = limit_price
            if order_type_lower in ['stop', 'stop_market']:
                order_type_lower = self._validate_and_prepare_stop_order(order_type_lower, stop_price, params)

            # Use CCXT's bracket order method if both TP and SL are provided (Bitget-specific)
            # This is the proper way to create bracket orders for both ONE_WAY and HEDGE modes
            # Note: In ONE_WAY mode, tradeSide should be omitted (params won't include it)
            if take_profit is not None and stop_loss is not None and not reduce_only:
                response = self.client.create_order_with_take_profit_and_stop_loss(
                    symbol=self.symbol,
                    type=order_type_lower,
                    side=side,
                    amount=quantity,
                    price=price,
                    takeProfit=take_profit,
                    stopLoss=stop_loss,
                    params=params  # May include tradeSide='open' only in HEDGE mode
                )

                # Extract order ID from response
                if isinstance(response, dict) and 'id' in response:
                    self.last_order_id = response['id']
                    logger.info(f"Bracket order executed: {side} {quantity} @ {order_type_lower} (TP={take_profit}, SL={stop_loss}, ID: {self.last_order_id})")
                else:
                    logger.info(f"Bracket order executed: {side} {quantity} @ {order_type_lower} (TP={take_profit}, SL={stop_loss})")
                logger.debug(f"Full bracket order response: {response}")

                return True

            # Otherwise, use standard order creation
            response = self.client.create_order(
                symbol=self.symbol,
                type=order_type_lower,
                side=side,
                amount=quantity,
                price=price,
                params=params
            )

            # Extract order ID from response
            if isinstance(response, dict) and 'id' in response:
                self.last_order_id = response['id']
                logger.info(f"Order executed: {side} {quantity} @ {order_type_lower} (ID: {self.last_order_id})")
            else:
                logger.info(f"Order executed: {side} {quantity} @ {order_type_lower}")
            logger.debug(f"Full order response: {response}")

            # Create take profit order separately if only TP is specified
            if take_profit is not None and stop_loss is None and not reduce_only:
                tp_side = self._get_opposite_side(side)
                tp_params = self._build_order_params(reduce_only=True)

                self.client.create_order(
                    symbol=self.symbol,
                    type='limit',
                    side=tp_side,
                    amount=quantity,
                    price=take_profit,
                    params=tp_params
                )
                logger.debug(f"Take profit order created at {take_profit}")

            # Create stop loss order separately if only SL is specified
            if stop_loss is not None and take_profit is None and not reduce_only:
                sl_side = self._get_opposite_side(side)
                sl_params = self._build_order_params(reduce_only=True)

                self.client.create_stop_market_order(
                    symbol=self.symbol,
                    side=sl_side,
                    amount=quantity,
                    stopPrice=stop_loss,
                    params=sl_params
                )
                logger.debug(f"Stop loss order created at {stop_loss}")

            return True

        except Exception as e:
            logger.error(f"Error executing trade: {str(e)}")
            return False

    def check_both_positions_open(self) -> bool:
        """
        Check if both long and short positions are open (hedge mode issue).

        Returns:
            True if both positions are open, False otherwise
        """
        try:
            positions = self.client.fetch_positions([self.symbol])
            long_open = False
            short_open = False

            for pos in positions:
                if pos['symbol'] == self.symbol:
                    contracts = float(pos.get('contracts', 0))
                    side = pos.get('side', 'long')

                    if contracts > 0:
                        if side == 'long':
                            long_open = True
                        elif side == 'short':
                            short_open = True

            if long_open and short_open:
                logger.warning("⚠️  Both long and short positions are open! Your account is in HEDGE mode.")
                logger.warning("    Please close all positions and switch to ONE_WAY mode.")
                return True

            return False
        except Exception as e:
            logger.error(f"Error checking positions: {e}")
            return False

    def get_status(self) -> Dict[str, Union[OrderStatus, PositionStatus, None]]:
        """
        Get current order and position status using CCXT.

        Returns:
            Dictionary containing order_status and position_status
        """
        status = {}

        try:
            # Get order status if we have a last order
            if self.last_order_id:
                order = self.client.fetch_order(self.last_order_id, self.symbol)
                if order:
                    status["order_status"] = OrderStatus(
                        is_open=order['status'] not in ['closed', 'canceled', 'expired'],
                        order_id=str(order.get('id', '')),
                        filled_qty=float(order.get('filled', 0)),
                        filled_avg_price=float(order.get('average', 0)),
                        status=order.get('status', 'unknown'),
                        side=order.get('side', 'unknown'),
                        order_type=order.get('type', 'unknown'),
                    )

            # Get position status using CCXT
            positions = self.client.fetch_positions([self.symbol])

            if positions and len(positions) > 0:
                # Find the position for this symbol
                pos = None
                for p in positions:
                    if p['symbol'] == self.symbol:
                        pos = p
                        break

                if pos and float(pos.get('contracts', 0)) != 0:
                    # CCXT returns contracts as the position size
                    contracts = float(pos.get('contracts', 0))
                    side = pos.get('side', 'long')  # 'long' or 'short'

                    # Convert to signed quantity (positive for long, negative for short)
                    qty = contracts if side == 'long' else -contracts

                    entry_price = float(pos.get('entryPrice', 0))
                    mark_price = float(pos.get('markPrice', entry_price))
                    unrealized_pnl = float(pos.get('unrealizedPnl', 0))
                    unrealized_pnl_pct = self._calculate_unrealized_pnl_pct(qty, entry_price, mark_price)

                    status["position_status"] = PositionStatus(
                        qty=qty,
                        notional_value=float(pos.get('notional', abs(contracts * mark_price))),
                        entry_price=entry_price,
                        unrealized_pnl=unrealized_pnl,
                        unrealized_pnl_pct=unrealized_pnl_pct,
                        mark_price=mark_price,
                        leverage=int(pos.get('leverage', self.leverage)),
                        margin_mode=pos.get('marginMode', self.margin_mode.value),
                        liquidation_price=float(pos.get('liquidationPrice', 0)),
                    )
                else:
                    status["position_status"] = None
            else:
                status["position_status"] = None

        except Exception as e:
            logger.error(f"Error getting status: {str(e)}")
            status["position_status"] = None

        return status

    def get_account_balance(self) -> Dict[str, float]:
        """
        Get futures account balance using CCXT.

        Returns:
            Dictionary with balance information

        Raises:
            RuntimeError: If balance cannot be retrieved
        """
        try:
            # Fetch balance using CCXT - it uses the defaultType (swap) automatically
            balance = self.client.fetch_balance()

            logger.debug(f"Raw balance response: {balance}")

            # CCXT returns balance with 'total', 'free', 'used' for each currency
            # For futures, we need to look at the USDT balance
            usdt_balance = balance.get('USDT', {})

            logger.debug(f"USDT balance: {usdt_balance}")

            # Get total equity (includes unrealized PnL)
            total = float(usdt_balance.get('total', 0))
            free = float(usdt_balance.get('free', 0))

            # Unrealized PnL can be calculated from positions
            unrealized_pnl = 0.0
            try:
                positions = self.client.fetch_positions([self.symbol])
                for pos in positions:
                    if pos['symbol'] == self.symbol:
                        unrealized_pnl += float(pos.get('unrealizedPnl', 0))
            except:
                pass

            result = {
                "total_wallet_balance": total,
                "available_balance": free,
                "total_unrealized_profit": unrealized_pnl,
                "total_margin_balance": total,
            }

            logger.debug(f"Account balance: total={total:.2f}, available={free:.2f}, pnl={unrealized_pnl:.4f}")

            # Warn if balance is zero in demo mode
            if self.demo and total == 0:
                logger.warning(
                    "⚠️  Demo account balance is 0 USDT! "
                    "Please fund your Bitget demo account at: "
                    "https://www.bitget.com/demo-trading"
                )

            return result

        except Exception as e:
            logger.error(f"Error getting account balance: {str(e)}")
            raise RuntimeError(f"Failed to get account balance: {e}") from e

    def get_mark_price(self) -> float:
        """
        Get current mark price for the symbol using CCXT.

        Returns:
            Current mark price

        Raises:
            RuntimeError: If mark price cannot be retrieved
        """
        try:
            # Fetch ticker using CCXT
            ticker = self.client.fetch_ticker(self.symbol)

            # Try to get mark price, fall back to last price
            mark_price = ticker.get('info', {}).get('markPrice')
            if mark_price:
                return float(mark_price)

            # Fallback to last price if mark price not available
            return float(ticker.get('last', 0))

        except Exception as e:
            logger.error(f"Error getting mark price: {str(e)}")
            raise RuntimeError(f"Failed to get mark price: {e}") from e

    def get_open_orders(self) -> List[Dict]:
        """Get all open orders for the symbol using CCXT."""
        try:
            orders = self.client.fetch_open_orders(self.symbol)
            return orders if orders else []
        except Exception as e:
            logger.error(f"Error getting open orders: {str(e)}")
            return []

    def cancel_open_orders(self) -> bool:
        """Cancel all open orders for the symbol using CCXT."""
        try:
            # Get all open orders first
            open_orders = self.get_open_orders()

            # Cancel each order
            for order in open_orders:
                try:
                    self.client.cancel_order(order['id'], self.symbol)
                except Exception as e:
                    logger.warning(f"Failed to cancel order {order['id']}: {e}")

            if len(open_orders) > 0:
                logger.info(f"Cancelled {len(open_orders)} open orders")
            else:
                logger.debug("No open orders to cancel")
            return True
        except Exception as e:
            logger.error(f"Error cancelling open orders: {str(e)}")
            return False

    def close_position(self, position_side: Optional[str] = None) -> bool:
        """
        Close the current position using CCXT.

        Args:
            position_side: Optional position side for hedge mode

        Returns:
            bool: True if position was closed successfully
        """
        try:
            status = self.get_status()
            position = status.get("position_status")

            if position is None or position.qty == 0:
                logger.debug("No open position to close")
                return True

            # Determine side to close
            qty = abs(position.qty)
            position_side_str = "buy" if position.qty > 0 else "sell"
            side = self._get_opposite_side(position_side_str)

            # Create market order to close position
            params = self._build_order_params(reduce_only=True)

            self.client.create_order(
                symbol=self.symbol,
                type='market',
                side=side,
                amount=qty,
                params=params
            )

            logger.info(f"Position closed: {qty} {side}")
            return True

        except Exception as e:
            # "No position to close" error is expected and not really an error
            if self._is_no_position_error(e):
                logger.debug("No open position to close")
                return True  # Not an error, just no position
            else:
                logger.error(f"Error closing position: {str(e)}")
                return False

    def set_leverage(self, leverage: int) -> bool:
        """
        Change leverage for the symbol using CCXT.

        Args:
            leverage: New leverage value (1-125)

        Returns:
            bool: True if successful
        """
        try:
            params = {
                'marginCoin': self.margin_coin,
            }
            self.client.set_leverage(leverage, self.symbol, params=params)
            self.leverage = leverage
            logger.debug(f"Leverage set to {leverage}x for {self.symbol}")
            return True
        except Exception as e:
            logger.error(f"Error setting leverage: {str(e)}")
            return False

    def set_margin_mode(self, mode: MarginMode) -> bool:
        """
        Change margin mode for the symbol using CCXT.

        Args:
            mode: New margin mode (ISOLATED or CROSSED)

        Returns:
            bool: True if successful
        """
        try:
            self.client.set_margin_mode(mode.to_ccxt(), self.symbol)
            self.margin_mode = mode
            logger.info(f"Margin mode set to {mode.value} for {self.symbol}")
            return True
        except Exception as e:
            logger.error(f"Error setting margin mode: {str(e)}")
            return False


# Example usage
if __name__ == "__main__":
    import os

    # Initialize with demo mode
    print("Initializing BitgetFuturesOrderClass with CCXT...")

    try:
        order_manager = BitgetFuturesOrderClass(
            symbol="BTC/USDT:USDT",  # CCXT perpetual swap format
            trade_mode=TradeMode.QUANTITY,
            api_key=os.getenv("BITGETACCESSAPIKEY", ""),
            api_secret=os.getenv("BITGETSECRETKEY", ""),
            passphrase=os.getenv("BITGETPASSPHRASE", ""),
            demo=True,
            leverage=5,
        )

        print("✓ Initialized successfully with CCXT")

        # Get account balance
        print("\nGetting account balance...")
        balance = order_manager.get_account_balance()
        print(f"Account balance: {balance}")

        # Get mark price
        print("\nGetting mark price...")
        price = order_manager.get_mark_price()
        print(f"Mark price: {price}")

        # Get status
        print("\nGetting status...")
        status = order_manager.get_status()
        print(f"Status: {status}")

        print("\n✅ All operations completed successfully!")

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
