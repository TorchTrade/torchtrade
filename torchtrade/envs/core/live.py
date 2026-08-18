"""Base class for live trading environments."""

import logging
import math
import time
from abc import abstractmethod
from datetime import datetime, timedelta
from enum import Enum
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import torch
from tensordict import TensorDictBase
from torchrl.data import Unbounded

from torchtrade.envs.core.base import TorchTradeBaseEnv
from torchtrade.envs.core.state import (
    PositionState,
    position_direction_from_status,
    POSITION_DUST_EPS,
    position_qty_from_status,
)
from torchtrade.envs.utils.timeframe import timeframe_to_seconds
from torchtrade.envs.utils.termination import is_bankrupt


logger = logging.getLogger(__name__)


class ObservationFailurePolicy(str, Enum):
    """What a live env does with an open position when it can no longer read the venue."""

    HALT = "halt"
    FLATTEN = "flatten"


class LiveObservationHalt(RuntimeError):
    """Raised when a live env cannot read its own account state.

    `flatten_accepted` records that the venue accepted a close REQUEST, not that the
    position is gone. See docs/environments/online.md.
    """

    def __init__(self, error, policy, flatten_accepted=None, flatten_error=None):
        self.original_exception = error
        self.policy = policy
        self.flatten_accepted = flatten_accepted
        self.flatten_error = flatten_error
        super().__init__(
            f"live state acquisition failed with {type(error).__name__}: {error}"
        )


class TorchTradeLiveEnv(TorchTradeBaseEnv):
    """
    Base class for live trading environments.

    Provides common functionality for all live trading environments:
    - Observer/trader dependency injection pattern
    - Common waiting logic (_wait_for_next_timestamp)
    - Market data observation spec construction pattern
    - Reset scaffolding
    - Bankruptcy termination check (_check_termination)

    Subclasses must implement:
    - _init_trading_clients(): Provider-specific client initialization
    - _get_portfolio_value(): Provider-specific portfolio calculation
    - _step(): Environment step logic
    """

    def __init__(
        self,
        config,
        api_key: str = "",
        api_secret: str = "",
        observer=None,
        trader=None,
        timezone: str = "America/New_York",
    ):
        """
        Initialize live trading environment.

        Args:
            config: Environment configuration
            api_key: API key for trading provider
            api_secret: API secret for trading provider
            observer: Optional pre-configured observation client (dependency injection)
            trader: Optional pre-configured trading client (dependency injection)
            timezone: Timezone for time-based operations (default: America/New_York)
        """
        # Initialize base class first
        super().__init__(config)

        # Store timezone with validation
        try:
            self.timezone = ZoneInfo(timezone)
        except ZoneInfoNotFoundError as e:
            raise ValueError(
                f"Invalid timezone: '{timezone}'. Common valid timezones include: "
                f"'America/New_York', 'America/Chicago', 'America/Los_Angeles', 'UTC', "
                f"'Europe/London', 'Asia/Tokyo'. See "
                f"https://en.wikipedia.org/wiki/List_of_tz_database_time_zones "
                f"for a complete list of valid timezone names."
            ) from e

        # Store execution timeframe
        # Note: Subclasses should ensure execute_on has 'value' and 'unit' attributes
        # For TimeFrame objects: execute_on.value (int) and execute_on.unit (TimeFrameUnit enum)
        self.execute_on = None

        # Initialize trading clients (observer and trader)
        # Subclasses should override _init_trading_clients to provide provider-specific setup
        self._init_trading_clients(api_key, api_secret, observer, trader)

        # Initialize position state
        # Note: Subclasses may override this with their specific position tracking needs
        self.position = PositionState()

    @abstractmethod
    def _init_trading_clients(self, api_key: str, api_secret: str, observer, trader):
        """
        Initialize observer and trader clients.

        Subclasses must implement this to provide provider-specific initialization:
        - Use injected observer/trader if provided
        - Otherwise create new instances with api_key/api_secret

        Should set:
        - self.observer: Observation/data client
        - self.trader: Order execution client

        Args:
            api_key: API key for provider
            api_secret: API secret for provider
            observer: Optional pre-configured observer
            trader: Optional pre-configured trader
        """
        raise NotImplementedError("Subclasses must implement _init_trading_clients()")

    def _wait_for_next_timestamp(self):
        """
        Wait until next time step using single calculated sleep.

        Uses a single sleep call with exact duration instead of a polling loop,
        improving CPU efficiency and timing precision.

        This is COMMON across all live environments - timing logic is universal.

        Uses:
        - self.execute_on: the execution TimeFrame
        - self.timezone: Timezone for time calculations
        """
        # One canonical duration, from the TimeFrame itself. This used to look the
        # unit up in a 17-entry alias map -- "TimeFrameUnit.Minute", "Minute", "Min",
        # "min", "minute", "h", "H", "D", "d", "seconds" -- because five exchanges
        # stringified the same enum four different ways (`str(unit)`, `str(unit.value)`,
        # a hardcoded "seconds") and the map grew an entry per spelling rather than the
        # spellings being fixed. A sixth exchange spelling it a fifth way would have
        # raised at the first bar, in production, on a timer (#288).
        wait_duration = timedelta(seconds=timeframe_to_seconds(self.execute_on))

        # Calculate next step time
        current_time = datetime.now(self.timezone)
        next_step = (current_time + wait_duration).replace(second=0, microsecond=0)

        # Calculate exact sleep duration
        sleep_seconds = (next_step - datetime.now(self.timezone)).total_seconds()

        # Single sleep instead of polling loop
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)

    def _sync_action_level_after_reset(self) -> None:
        """Reconcile current_action_level with the position observed during reset.

        current_action_level is only assigned after a trade THIS env made, so a position
        the env did not open (pre-existing on the exchange) leaves it at the stale 0.0
        default -- and _execute_trade_if_needed's ``desired_action == current_action_level``
        guard would then short-circuit a flat command and never close it.

        The level that produced such a position is unknowable: NaN never compares equal,
        so the next command always executes.

        NOT subsumed by _sync_position_from_exchange, though it looks like it should be:
        _reset writes current_position from the exchange FIRST, so by the first _step the
        cache and the exchange already agree -- and a sync that only acts on a MISMATCH sees
        none, and leaves the stale level alone. Reset answers "is this position mine?"; step
        answers "is it still there?". Only reset can know the position predates the episode.
        """
        self.position.current_action_level = (
            0.0 if self.position.current_position == 0 else float("nan")
        )
        # And the size target, which no live _reset cleared: a fresh episode inherited the
        # previous one's, so the first sync compared a flat account against last episode's
        # position and released the guard it had just computed -- warning about a
        # discrepancy that did not exist.
        self.position.target_qty = None
        self.position.target_tol = 0.0
        self.position.target_reported = False

    def _sync_position_from_exchange(self, position_status) -> None:
        """Overwrite the cached position with what the exchange actually holds.

        Call at the top of _step(), BEFORE the duplicate-action guard.

        current_position/current_action_level are written only by trades THIS env made, so a
        liquidation or a manual close leaves them stale, and _execute_trade_if_needed's
        ``desired_action == current_action_level`` guard then refuses the agent's re-entry for
        the rest of the episode.

        On a match the level is left alone -- that is what still lets the guard suppress a
        genuinely redundant trade. The level behind a position we did not open is unknowable,
        so it becomes NaN, which never compares equal.

        A mismatch also discards hold_counter. The position we were counting is gone (or was
        never ours), and this is the only code that knows that -- so if the agent re-enters in
        THIS same step, the new position must start from zero rather than inherit the dead
        one's age.

        Size is reconciled as well as direction. A partial fill leaves the DIRECTION
        intact, so the check above cannot see it -- the env would believe it holds the
        level it asked for while holding something else, and the guard would suppress
        every corrective retry, permanently and silently. The tolerance is the venue's own
        minimum tradeable size, carried on the trade: below that there is no order that
        could correct the difference, so firing on it would never converge.

        An unknown status raises rather than syncing to the 0 an unreachable exchange
        would produce. In practice _step raises earlier, on its own status read.
        """
        observed = position_direction_from_status(position_status)

        if observed != self.position.current_position:
            self.position.current_action_level = 0.0 if observed == 0 else float("nan")
            self.position.hold_counter = 0
            self.position.target_qty = None
            self.position.target_tol = 0.0
            self.position.target_reported = False
        elif self.position.target_qty is not None and (
            abs(position_qty_from_status(position_status) - self.position.target_qty)
            >= max(self.position.target_tol, POSITION_DUST_EPS)
        ):
            # An explicit flag, not `isnan(level)`: reset writes NaN to mean "this position
            # predates the episode and its level is unknowable", so reusing NaN as the
            # already-reported marker silenced every divergence in such an episode.
            if not self.position.target_reported:
                logger.warning(
                    "venue holds %s but the last action asked for %s; releasing the "
                    "duplicate-action guard so the agent can correct it",
                    position_qty_from_status(position_status), self.position.target_qty,
                )
            self.position.target_reported = True
            self.position.current_action_level = float("nan")

        self.position.current_position = observed

    def _declare_base_features_spec(self, base_window: int) -> None:
        """Declare the base_features spec (raw OHLC, first timeframe) -- shape (base_window, 4).

        _get_observation emits base_features when include_base_features is set, so every live
        env's _build_observation_specs MUST call this or observation_spec disagrees with the
        emitted observation: check_env_specs fails and a collector pre-allocating from the spec
        silently drops the key. Shared here alongside the emitter so the spec cannot drift
        per-exchange -- the exact drift that left 3 of 4 futures exchanges undeclared (#61).
        """
        if self.config.include_base_features:
            self.observation_spec.set(
                "base_features",
                Unbounded(shape=(base_window, 4), dtype=torch.float),
            )

    def _capture_bankruptcy_baseline(self) -> None:
        """Record the equity an episode starts from, or refuse to start (#345).

        `is_bankrupt` is `current < threshold * initial`, so a baseline of 0 never fires,
        and a NaN baseline makes the comparison False forever -- the check is then off for
        the whole episode. `nan <= 0` is False, so isfinite has to carry that half.
        """
        self.initial_portfolio_value = self._get_portfolio_value()
        if not math.isfinite(self.initial_portfolio_value) or self.initial_portfolio_value <= 0:
            raise ValueError(
                f"cannot start an episode on equity of {self.initial_portfolio_value}"
            )

    def _record_position_after_trade(self, desired_action: float, trade_info: dict) -> None:
        """Record the position the trade RESULTED in, not the side that was sent (#276).

        Under fractional sizing a SELL that only trims a long leaves a long; recording it
        as a short makes the next bar's sync see a mismatch the env inflicted on itself.

        Takes the whole `trade_info` rather than the two size values separately: passed
        separately, all five call sites forwarded the target and dropped the tolerance,
        leaving it 0.0 so the next bar called every COMPLETE fill a divergence. One
        argument cannot be half-passed.
        """
        if not trade_info.get("executed") or trade_info.get("success") is False:
            # A refusal that means "we are already there" must RE-ARM the guard. Without
            # it, a release latches: the release compares against the snapshot target, the
            # refusal against one recomputed from drifting equity, so the two disagree and
            # nothing ever restores a finite level -- the env then re-sizes every bar and
            # trades whenever drift clears the venue minimum, unrequested and unlogged.
            if trade_info.get("at_target"):
                self.position.current_action_level = desired_action
                self.position.target_qty = None
                self.position.target_tol = 0.0
                self.position.target_reported = False
            return

        tol = trade_info.get("target_tol") or 0.0
        self.position.current_position = (desired_action > 0) - (desired_action < 0)
        self.position.current_action_level = desired_action
        self.position.target_qty = trade_info.get("target_qty")
        # A venue reporting min_qty as NaN would make max(nan, eps) NaN, and `x >= nan` is
        # False -- the check would be off with nothing to show for it. CCXT reports 0 here
        # routinely, which is why this falls back rather than raising.
        self.position.target_tol = tol if math.isfinite(tol) and tol > 0 else 0.0
        self.position.target_reported = False

    def _check_termination(self, portfolio_value: float) -> bool:
        """Terminate when the portfolio falls below bankrupt_threshold * its initial value."""
        return is_bankrupt(
            current=portfolio_value,
            initial=self.initial_portfolio_value,
            threshold=self.config.bankrupt_threshold,
            enabled=self.config.done_on_bankruptcy,
        )

    @abstractmethod
    def _get_portfolio_value(self, *args, **kwargs) -> float:
        """
        Calculate total portfolio value.

        MUST be implemented by subclasses as calculation is provider-specific.

        Examples:
        - Alpaca spot: cash + position_market_value
        - Binance futures: total_margin_balance
        - Interactive Brokers: net_liquidation_value

        Returns:
            Total portfolio value (float)
        """
        raise NotImplementedError(
            "Subclasses must implement _get_portfolio_value() "
            "as portfolio calculation is provider-specific"
        )

    @abstractmethod
    def _step(self, tensordict: TensorDictBase) -> TensorDictBase:
        """
        Execute one environment step.

        MUST be implemented by subclasses for provider-specific step logic.

        Args:
            tensordict: Input tensordict containing action

        Returns:
            TensorDict with next observation, reward, done flags
        """
        raise NotImplementedError(
            "Subclasses must implement _step() for environment step logic"
        )
