"""Base class for Alpaca live trading environments."""

import logging
import math
from abc import abstractmethod
from typing import Callable, Optional

import torch
from tensordict import TensorDict, TensorDictBase

from torchtrade.envs.live.alpaca.observation import AlpacaObservationClass
from torchtrade.envs.live.alpaca.order_executor import AlpacaOrderClass
from torchtrade.envs.core.live import TorchTradeLiveEnv
from torchtrade.envs.core.live import STATUS_UNKNOWN_KEY
from torchtrade.envs.core.state import (
    HistoryTracker,
    PositionState,
    advance_hold_counter,
    position_direction_from_status,
)


logger = logging.getLogger(__name__)


class AlpacaBaseTorchTradingEnv(TorchTradeLiveEnv):
    """
    Base class for Alpaca trading environments.

    Provides common functionality for all Alpaca environments:
    - AlpacaObservationClass and AlpacaOrderClass initialization
    - Observation spec construction (account state + market data)
    - Common observation gathering logic
    - Portfolio value calculation (cash + position_market_value)
    - Helper methods for market data keys and account state

    Standard account state for Alpaca environments (6 elements):
    [exposure_pct, position_direction, unrealized_pnl_pct,
     holding_time, leverage, distance_to_liquidation]

    Element definitions:
        - exposure_pct: position_value / portfolio_value (0.0 to 1.0 for spot)
        - position_direction: sign(position_size). Spot cannot short, so this is 0 or +1
          in practice -- but it is NOT clamped: a negative qty from the broker is
          reported as -1 rather than laundered into 'flat', so the observation always
          agrees with the position state the env tracks internally.
        - unrealized_pnl_pct: (current_price - entry_price) / entry_price * direction
        - holding_time: steps since position opened
        - leverage: Always 1.0 for spot (no leverage)
        - distance_to_liquidation: Always 1.0 for spot (no liquidation risk)

    Subclasses must implement:
    - Action space definition (different for standard vs SLTP)
    - _execute_trade_if_needed(): Trade execution logic
    - _calculate_trade_amount(): Trade sizing logic
    """

    # Standard account state for Alpaca environments (6 elements)
    # Universal state used across all TorchTrade environments for better generalization.
    ACCOUNT_STATE = [
        "exposure_pct", "position_direction", "unrealized_pnlpct",
        "holding_time", "leverage", "distance_to_liquidation"
    ]

    def __init__(
        self,
        config,
        api_key: str = "",
        api_secret: str = "",
        feature_preprocessing_fn: Optional[Callable] = None,
        observer: Optional[AlpacaObservationClass] = None,
        trader: Optional[AlpacaOrderClass] = None,
    ):
        """
        Initialize Alpaca trading environment.

        Args:
            config: Environment configuration
            api_key: Alpaca API key (not required if observer and trader are provided)
            api_secret: Alpaca API secret (not required if observer and trader are provided)
            feature_preprocessing_fn: Optional custom preprocessing function
            observer: Optional pre-configured AlpacaObservationClass for dependency injection
            trader: Optional pre-configured AlpacaOrderClass for dependency injection
        """
        # Store feature preprocessing function for use in _init_trading_clients
        self._feature_preprocessing_fn = feature_preprocessing_fn

        # Initialize base class (will call _init_trading_clients)
        super().__init__(
            config=config,
            api_key=api_key,
            api_secret=api_secret,
            observer=observer,
            trader=trader,
            timezone="America/New_York"
        )

        # Extract execute_on timeframe
        self.execute_on = config.execute_on

        # Reset settings
        # Cancel BEFORE closing, as all four futures envs do. The old order was
        # close-then-cancel, and cancel_open_orders() falls through to the account-wide
        # cancel-all -- so it cancelled the market close it had just submitted, which
        # close_all_positions() does not block on. Outside RTH the close could not fill
        # first, and the init flatten was silently reverted (#289).
        self.trader.cancel_open_orders()
        if config.close_position_on_init:
            # close_position(), not close_all_positions(): the account-wide call iterates
            # get_all_positions() over EVERY symbol, and this flag defaults to True -- so
            # merely constructing an env flattened unrelated holdings. All four futures
            # envs are symbol-scoped at init too (#289).
            self.trader.close_position()

        # The env's own measure, not account.cash. The close above (when enabled) submits
        # market orders and does not block, so cash at this instant excludes whatever is
        # still tied up in an unsettled position: constructed holding $9k of BTC against
        # $1k cash, the baseline pinned at 1000 and _check_termination then fired below
        # $100 rather than $1000 (#284).
        #
        # Using _get_portfolio_value rather than account.portfolio_value keeps the
        # baseline the SAME quantity that termination compares against -- cash plus
        # position value -- so the two cannot drift apart. The futures bases already read
        # an equity figure for this reason.
        self._capture_bankruptcy_baseline()

        # Build observation specs
        self._build_observation_specs()

        # Initialize position state
        self.position = PositionState()

        # Initialize history tracking
        self.history = HistoryTracker()

    def _init_trading_clients(
        self,
        api_key: str,
        api_secret: str,
        observer: Optional[AlpacaObservationClass],
        trader: Optional[AlpacaOrderClass]
    ):
        """
        Initialize Alpaca observer and trader clients.

        Uses dependency injection pattern - uses provided instances or creates new ones.
        """
        # Initialize observer
        self.observer = observer if observer is not None else AlpacaObservationClass(
            symbol=self.config.symbol,
            timeframes=self.config.time_frames,
            window_sizes=self.config.window_sizes,
            feature_preprocessing_fn=self._feature_preprocessing_fn,
        )

        # Initialize trader
        self.trader = trader if trader is not None else AlpacaOrderClass(
            symbol=self.config.symbol.replace('/', ''),
            trade_mode=self.config.trade_mode,
            api_key=api_key,
            api_secret=api_secret,
            paper=self.config.paper,
        )

    def _get_observation(self, advance_hold: bool = True) -> TensorDictBase:
        """Get the current observation state.

        Args:
            advance_hold: If True (the default, used by `_step()`), ages `hold_counter`
                by one bar using the direction observed in THIS method's single
                `get_status()` call -- holding_time and position_direction in the
                emitted account_state are always derived from the same snapshot.
                `_reset()` passes False so a reset can never itself count a bar.
        """
        # Get market data
        obs_dict = self.observer.get_observations(
            return_base_ohlc=self.config.include_base_features
        )

        # Extract base features if requested
        if self.config.include_base_features:
            base_features = obs_dict.get("base_features")

        # Get market data for each timeframe
        market_data = [obs_dict[features_name] for features_name in self.observer.get_keys()]

        # Get account state from trader (single fetch: holding_time and
        # position_direction below MUST come from this same snapshot)
        status = self.trader.get_status()
        account = self.trader.client.get_account()
        cash = float(account.cash)
        # cash IS alpaca's equity: it is the whole of portfolio_value when flat, and the
        # bankruptcy baseline and every reward derive from it. A NaN passes the
        # held-position guard below (which is keyed on direction) and reaches
        # is_bankrupt(), where `nan < threshold * initial` is False -- termination off
        # for the episode. `not isfinite` and not `<= 0`, since +inf passes that too.
        if not math.isfinite(cash):
            raise ValueError(
                f"venue reported a non-finite cash balance ({cash}); refusing to derive "
                f"an account state or a bankruptcy check from it"
            )
        position_status = status.get("position_status", None)

        # Calculate portfolio value
        # Dust is not a position: gating on `is None` let a 1e-12 residual left behind a
        # close take the position branch and read stale fields off it.
        position_direction = float(position_direction_from_status(position_status))
        if advance_hold:
            advance_hold_counter(self.position, position_direction)
        holding_time = float(self.position.hold_counter)

        if position_direction == 0:
            position_value = 0.0
            entry_price = 0.0
            unrealized_pnlpc = 0.0
            portfolio_value = cash
            # Get current market price even when no position
            try:
                current_price = self.observer.get_current_price()
            except Exception:
                current_price = 0.0
        else:
            # Same finiteness contract as the futures envs (#277). market_value and
            # unrealized_plpc reach account_state directly; avg_entry_price and
            # current_price are read but unused here, and are checked anyway so the
            # contract does not depend on which locals happen to be dead. NaN passes
            # every comparison below
            # -- a NaN market_value reads as a flat account holding a position, a NaN
            # unrealized_plpc goes into the tensor and on into the policy network. Spot is
            # not exempt from invariant #3.
            for _name, _value in (
                ("qty", position_status.qty),
                ("market_value", position_status.market_value),
                ("avg_entry_price", position_status.avg_entry_price),
                ("current_price", position_status.current_price),
                ("unrealized_plpc", position_status.unrealized_plpc),
            ):
                if not math.isfinite(float(_value)):
                    raise ValueError(
                        f"venue reported a non-finite {_name} ({_value}) for an open "
                        f"position; refusing to derive an account state from it"
                    )
            position_value = position_status.market_value
            entry_price = position_status.avg_entry_price
            current_price = position_status.current_price
            unrealized_pnlpc = position_status.unrealized_plpc
            portfolio_value = cash + position_value

        # Calculate new 6-element account state
        # Element 0: exposure_pct (position_value / portfolio_value)
        # A position held against a non-positive portfolio value has no exposure_pct to
        # report -- the ratio is unbounded -- and `else 0.0` reports the one value that is
        # certainly wrong: a flat-looking account that is holding a position (#277).
        # `not (x > 0)` because NaN compares False to everything.
        if not (portfolio_value > 0) and position_direction != 0:
            raise ValueError(
                f"Position worth {position_value} held against a non-positive portfolio "
                f"value ({portfolio_value}); refusing to report this account as flat."
            )
        exposure_pct = position_value / portfolio_value if portfolio_value > 0 else 0.0

        # Element 2: unrealized_pnl_pct (inherited from Alpaca)
        # Element 3: holding_time
        # Element 4: leverage (always 1.0 for spot)
        leverage = 1.0

        # Element 5: distance_to_liquidation (always 1.0 for spot, no liquidation)
        distance_to_liquidation = 1.0

        # Build 6-element account state tensor
        # [exposure_pct, position_direction, unrealized_pnl_pct,
        #  holding_time, leverage, distance_to_liquidation]
        account_state = torch.tensor(
            [exposure_pct, position_direction, unrealized_pnlpc, holding_time, leverage, distance_to_liquidation],
            dtype=torch.float
        )

        # Build output TensorDict
        out_td = TensorDict(
            {
                self.account_state_key: account_state,
                # Always 0.0: `_finalize_step_flags` sets it per step (#295).
                STATUS_UNKNOWN_KEY: torch.zeros(1, dtype=torch.float),
            },
            batch_size=(),
        )
        for market_data_name, data in zip(self.market_data_keys, market_data):
            out_td.set(market_data_name, torch.from_numpy(data))

        # Add base features if requested
        if self.config.include_base_features and base_features is not None:
            out_td.set("base_features", torch.from_numpy(base_features))

        return out_td

    def _read_cash(self) -> float:
        """The cash balance, validated at the read (#347).

        The SLTP env sizes orders straight off `self.balance`, and `_get_portfolio_value`
        -- which does check -- is a SEPARATE fetch its `_step` makes AFTER the order is
        already on the venue. Zero cash is legitimate (fully invested); negative is a
        margin debit that would size a negative-notional buy.
        """
        cash = float(self.trader.client.get_account().cash)
        if not math.isfinite(cash) or cash < 0:
            raise ValueError(f"venue reported an unusable cash balance ({cash})")
        return cash

    def _get_portfolio_value(self) -> float:
        """
        Calculate total portfolio value for Alpaca.

        Returns cash + position_market_value.

        Raises PositionUnknownError on an unknown status -- see the comment on the return.
        """
        status = self.trader.get_status()
        position_status = status.get("position_status", None)

        self.balance = self._read_cash()

        if position_status is None:
            portfolio_value = self.balance
        else:
            # An unknown status raises on the .market_value read rather than taking the flat
            # branch above: cash alone feeds _check_termination, and for a held position that
            # is most of the portfolio missing -- an outage would read as a near-total loss.
            portfolio_value = self.balance + position_status.market_value

        # Guarded here as well as in _get_observation, because this is a SEPARATE fetch and
        # it is the value that reaches _check_termination and the reward. alpaca's _step
        # calls it BEFORE building an observation, so a NaN here is recorded and rewarded
        # against before the observation guard could ever fire (#277).
        if not math.isfinite(portfolio_value):
            raise ValueError(
                f"venue reported a non-finite portfolio value ({portfolio_value}); "
                f"refusing to run a bankruptcy check or compute a reward against it"
            )
        return portfolio_value

    def _reset(self, tensordict: TensorDictBase, **kwargs) -> TensorDictBase:
        """Reset the environment."""
        self._reset_outage_state()
        # Before any read below -- see the bybit copy for why the order matters (#278).
        self.observer.reset()
        if not self.trader.cancel_open_orders():
            logger.warning(
                "cancel_open_orders failed during reset; proceeding with potentially stale orders"
            )
        if self.config.close_position_on_reset:
            # close_position(), not close_all_positions(): the latter iterates
            # get_all_positions() over the WHOLE account, so an opt-in reset flatten
            # would market-close a second env's symbol or a manual holding at every
            # episode boundary. The four futures envs are symbol-scoped (#289).
            closed = self.trader.close_position()
            # Re-read rather than trusting the bool: alpaca's close_position() wraps a
            # client call that RAISES when the symbol is already flat (code 40410000),
            # so it returns False for the state we wanted. Warning on that would fire at
            # every episode boundary and train the operator to ignore the real one.
            # bybit returns True when flat, which is why the copied pattern misfired.
            if not closed and position_direction_from_status(
                self.trader.get_status().get("position_status")
            ) != 0:
                logger.warning(
                    "close_position_on_reset failed for %s; the episode starts with a "
                    "position it asked to be rid of", self.config.symbol,
                )

        # Reset history tracking
        self.history.reset()

        self.balance = self._read_cash()

        status = self.trader.get_status()
        position_status = status.get("position_status")
        self.position.hold_counter = 0

        self.position.current_position = position_direction_from_status(position_status)

        self._sync_action_level_after_reset()

        # Get initial observation. advance_hold=False: hold_counter was just zeroed
        # above; a reset must never itself count a bar (see advance_hold docstring).
        return self._get_observation(advance_hold=False)

    @abstractmethod
    def _execute_trade_if_needed(self, action) -> dict:
        """
        Execute trade if position change is needed.

        Must be implemented by subclasses as trade logic differs
        (standard 3-action vs SLTP bracket orders).

        Args:
            action: Action to execute (format varies by subclass)

        Returns:
            Dict with trade execution details
        """
        raise NotImplementedError(
            "Subclasses must implement _execute_trade_if_needed()"
        )

    @abstractmethod
    def _calculate_trade_amount(self, side: str) -> float:
        """
        Calculate the dollar amount to trade.

        Must be implemented by subclasses as sizing logic may differ.

        Args:
            side: "buy" or "sell"

        Returns:
            Dollar amount to trade (float)
        """
        raise NotImplementedError(
            "Subclasses must implement _calculate_trade_amount()"
        )

    def close(self, *, raise_if_closed: bool = True):
        """Clean up resources.

        Cancels orders; does NOT flatten. All four futures envs only warn here and tell
        the caller to `close_position()` first, and flattening unconditionally on
        shutdown made `close_position_on_init=False` pointless -- the position it was
        meant to preserve was market-closed the moment the process exited cleanly (#289).
        `examples/rule_based/live.py`, `examples/llm/local/live.py` and
        `examples/llm/frontier/live.py` call `env.close()`; they now exit with the
        position open, which is what the futures envs have always done.

        Keyword-only, matching EnvBase.close: TransformedEnv forwards `raise_if_closed`.
        """
        # Guarded like the futures close(): teardown is where an exception replaces
        # whatever error you were actually trying to see.
        try:
            if not self.trader.cancel_open_orders():
                logger.warning(
                    "cancel_open_orders failed during close(); orders may remain open"
                )
        except Exception as e:
            logger.error(f"Failed to cancel open orders on close(): {e}")
        try:
            # The dust rule, not `is not None`: a 1e-12 residual is not a position
            # (invariant 1), and an unknown status must not be reported as a held one.
            # Guarded because this is a network read on the shutdown path.
            held = position_direction_from_status(
                self.trader.get_status().get("position_status")
            )
        except Exception:
            held = 0
        if held != 0:
            logger.warning(
                "%s still holds a position at close(); call trader.close_position() "
                "first if you want it flattened", self.config.symbol,
            )
        super().close(raise_if_closed=raise_if_closed)

    def _get_current_price(self, position_status=None) -> float:
        """Get current market price with fallback chain.

        Tries multiple sources in order:
        1. Position status (if provided or fetched)
        2. Trader's current_price attribute (for mocks)
        3. Observer's get_current_price() (fetches from market data)

        Args:
            position_status: Optional position status to avoid redundant queries

        Returns:
            Current price, or 0.0 if unavailable

        Raises:
            PositionUnknownError: the exchange did not report the position. Both alpaca
                envs stop here, before any trade is sized -- POSITION_UNKNOWN is truthy
                and raises on attribute access.

        Lives on the base rather than on AlpacaTorchTradingEnv because the SLTP env is a
        sibling, not a subclass, and used to read `position_status.current_price if
        position_status else 0.0` inline -- so every flat bar recorded a price of 0 in its
        history (#290).
        """
        # Note: a caller that already resolved the status to None still pays a second
        # get_status() here, because None cannot be told from "not supplied". Both alpaca
        # envs do that on a flat bar. Negligible against alpaca's rate limit and matched
        # by the non-SLTP env before this change, so left as-is rather than widening #290.
        if position_status is None:
            position_status = self.trader.get_status().get("position_status", None)

        current_price = position_status.current_price if position_status else 0.0

        # Advance on USABLE, not `<= 0` (#349): NaN compares False to every operator, so
        # `nan <= 0` skipped both fallbacks and returned the NaN as the price.
        def usable(price):
            return math.isfinite(price) and price > 0

        if not usable(current_price) and hasattr(self.trader, 'current_price'):
            current_price = self.trader.current_price

        if not usable(current_price):
            try:
                fetched = self.observer.get_current_price()
                if usable(fetched):
                    # Logged AFTER validating: this said "Fetched current price: nan" and
                    # then threw it away, so the only trace claimed the fetch worked.
                    logger.info(f"Fetched current price from market data: {fetched}")
                current_price = fetched
            except Exception as e:
                logger.warning(f"Could not fetch current price: {e}")

        if not usable(current_price):
            logger.error(
                f"No usable price: every source in the chain failed "
                f"(last value {current_price}); reporting unavailable"
            )
            return 0.0
        return current_price
