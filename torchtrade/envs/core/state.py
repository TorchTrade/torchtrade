"""State management dataclasses for TorchTrade environments."""

import math
from dataclasses import asdict, dataclass, field, fields
from typing import Dict, List, Optional, Union


# Quantities at or below this are dust, not a position. Exchanges can leave a float residual
# (e.g. 1e-12) behind a full close; misreading it as an open position freezes the live envs'
# duplicate-action guard and puts a position the agent does not hold into its observation.
POSITION_DUST_EPS = 1e-9


class PositionUnknownError(RuntimeError):
    """Raised when an unknown exchange status reaches code that needs a real direction."""


class _PositionUnknown:
    """The exchange did not report a position we can trust, which is not "flat".

    Covers both a call that failed and one the exchange answered with an error code.

    Truthy, by leaving __bool__ alone. Every live env reads the status behind a
    truthiness check -- ``if position_status:`` in the futures envs, and
    ``position_status.<field> if position_status else <fallback>`` throughout alpaca --
    so a falsy sentinel would quietly take the flat branch and substitute a fallback
    price instead of stopping.
    """

    _instance = None

    def __new__(cls):
        # Identity survives pickling and copying, which is what makes `is
        # POSITION_UNKNOWN` safe as the check at every call site.
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __getattr__(self, name):
        if name.startswith("__") and name.endswith("__"):
            # copy/pickle probe the instance for __deepcopy__ and friends. Answering
            # those with a trading error breaks copying for a question that was never
            # about the position.
            raise AttributeError(name)
        # This is the guard every live _step actually hits: all ten read a field off the
        # status (mark_price/current_price) before anything else looks at it. A bare
        # AttributeError would also fail closed; raising here says why.
        raise PositionUnknownError(
            f"cannot read {name!r}: the exchange did not report the position"
        )

    def __repr__(self):
        return "POSITION_UNKNOWN"


# Returned by get_status() when the exchange call failed or reported an error. Distinct
# from None, which is a confirmed-flat account: reading a failed call as flat is how a
# held position gets re-bought every bar of an outage.
POSITION_UNKNOWN = _PositionUnknown()


def position_direction_from_status(position_status) -> int:
    """The direction the exchange holds: -1 short, 0 flat, +1 long. None means flat.

    The single rule for the LIVE envs: their position syncs, their resets, and the
    account_state they show the agent all go through here. The offline envs still derive
    direction their own way -- out of scope here, and not covered by this.

    POSITION_UNKNOWN raises rather than returning a direction. A caller that has not been
    taught to handle an unreachable exchange must fail loudly, because the alternative it
    would otherwise fall into is 0, and reading an outage as flat is the whole bug.
    """
    if position_status is POSITION_UNKNOWN:
        raise PositionUnknownError(
            "exchange status is unknown; the caller must handle this rather than "
            "treating it as a direction"
        )
    qty = 0.0 if position_status is None else float(position_status.qty)
    # A NaN qty fails `abs(qty) <= eps` and then `qty > 0`, so it fell through to -1 and
    # read as a SHORT. This rule runs before any finiteness check the envs apply, and
    # alpaca's _step syncs and trades on the result before building an observation -- so a
    # fabricated -1 drives a trade on a long-only account (#277).
    if not math.isfinite(qty):
        raise ValueError(
            f"exchange reported a non-finite position quantity ({qty}); refusing to "
            f"read a direction from it"
        )
    if abs(qty) <= POSITION_DUST_EPS:
        return 0
    return 1 if qty > 0 else -1


def position_qty_from_status(position_status) -> float:
    """The signed size the exchange holds, with dust read as flat. None means flat.

    The size twin of position_direction_from_status, and it exists for the same reason:
    the account_state paths honoured the dust rule while the TRADE paths hand-rolled
    `position.qty if position is not None else 0.0`, so a 1e-12 residual made
    `abs(current_qty) > 0` true on a flat account. That called close_position() on nothing
    and still advanced current_action_level from a trade that never happened, freezing the
    duplicate-action guard (#283).

    Returning exactly 0.0 rather than the residual means every downstream `== 0`,
    `abs(...) > 0` and `< 0` test is correct without each one repeating the epsilon.

    POSITION_UNKNOWN raises, for the same reason it does there: an outage read as flat is
    the whole bug.
    """
    if position_status is POSITION_UNKNOWN:
        raise PositionUnknownError(
            "exchange status is unknown; the caller must handle this rather than "
            "treating it as a size"
        )
    qty = 0.0 if position_status is None else float(position_status.qty)
    if not math.isfinite(qty):
        raise ValueError(
            f"exchange reported a non-finite position quantity ({qty}); refusing to "
            f"read a size from it"
        )
    return 0.0 if abs(qty) <= POSITION_DUST_EPS else qty


def binarize_action_type(action_type: str) -> int:
    """Convert action type string to binarized action value.

    Args:
        action_type: Action type string ("buy"/"long" -> 1, "sell"/"short" -> -1, others -> 0)

    Returns:
        Binarized action value (-1, 0, or 1)

    Examples:
        >>> binarize_action_type("buy")
        1
        >>> binarize_action_type("long")
        1
        >>> binarize_action_type("sell")
        -1
        >>> binarize_action_type("short")
        -1
        >>> binarize_action_type("hold")
        0
        >>> binarize_action_type("liquidation")
        0
    """
    if action_type in ("buy", "long"):
        return 1
    elif action_type in ("sell", "short"):
        return -1
    return 0


@dataclass
class PositionState:
    """Encapsulates all position-related state.

    Attributes:
        current_position: Current position indicator (0=no position, 1=long, -1=short)
        position_size: Number of units held (can be negative for shorts)
        position_value: Current market value of the position
        entry_price: Price at which the position was entered
        unrealized_pnlpc: Unrealized profit/loss as percentage of entry price
        hold_counter: Number of steps the position has been held
        hold_direction: The direction hold_counter is counting, so a direct flip
            (long -> short, never passing through flat) can be told apart from a hold
    """
    current_position: float = 0.0
    position_size: float = 0.0
    position_value: float = 0.0
    entry_price: float = 0.0
    unrealized_pnlpc: float = 0.0
    hold_counter: int = 0
    hold_direction: float = 0.0
    current_action_level: float = 0.0
    # What the last accepted action asked for, and the smallest divergence from it worth
    # acting on -- the venue's own minimum tradeable size, because anything below that
    # cannot be corrected by placing an order. None when the env has not traded.
    target_qty: Optional[float] = None
    target_tol: float = 0.0

    def reset(self):
        """Reset all position state to initial values."""
        for field in fields(self):
            setattr(self, field.name, field.default)


@dataclass
class HistoryTracker:
    """Encapsulates episode history tracking for all environments.

    Tracks price, action, reward, portfolio value, and position history across an episode.
    Provides methods to record steps, reset history, and export data for analysis.

    Attributes:
        base_prices: List of asset base prices at each step
        actions: List of actions taken at each step
        rewards: List of rewards received at each step
        portfolio_values: List of total portfolio values at each step
        positions: List of position sizes at each step (positive=long, negative=short, 0=flat)
        action_types: List of action types at each step ("hold", "buy", "sell", "long", "short", etc.)

    Example:
        >>> history = HistoryTracker()
        >>> history.record_step(price=50000.0, action=1.0, reward=0.05, portfolio_value=5000.0, position=0.5, action_type="long")
        >>> history.to_dict()
        {'base_prices': [50000.0], 'actions': [1.0], 'rewards': [0.05], 'portfolio_values': [5000.0], 'positions': [0.5], 'action_types': ['long']}
    """

    base_prices: List[float] = field(default_factory=list)
    actions: List[float] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)
    portfolio_values: List[float] = field(default_factory=list)
    positions: List[float] = field(default_factory=list)
    action_types: List[str] = field(default_factory=list)

    def reset(self) -> None:
        """Clear all history arrays.

        Resets the tracker to empty state, typically called at the start of a new episode.
        """
        self.base_prices.clear()
        self.actions.clear()
        self.rewards.clear()
        self.portfolio_values.clear()
        self.positions.clear()
        self.action_types.clear()

    def record_step(
        self,
        price: Union[int, float],
        action: Union[int, float],
        reward: Union[int, float],
        portfolio_value: Union[int, float],
        position: Union[int, float] = 0.0,
        action_type: str = "hold"
    ) -> None:
        """Record a single step's history.

        Args:
            price: Base asset price at this step
            action: Action taken at this step
            reward: Reward received at this step
            portfolio_value: Total portfolio value at this step
            position: Position size at this step (positive=long, negative=short, 0=flat). Defaults to 0.0.
            action_type: Type of action taken ("hold", "buy", "sell", "long", "short", etc.)
        """
        self.base_prices.append(price)
        self.actions.append(action)
        self.rewards.append(reward)
        self.portfolio_values.append(portfolio_value)
        self.positions.append(position)
        self.action_types.append(action_type)

    def to_dict(self) -> Dict[str, List[float]]:
        """Export history as dictionary for plotting/analysis.

        Returns:
            Dictionary mapping history field names to their value lists
        """
        return asdict(self)

    def __len__(self) -> int:
        """Return the number of steps recorded.

        Returns:
            Number of steps in the history
        """
        return len(self.base_prices)


def advance_hold_counter_from_size(position: PositionState) -> None:
    """advance_hold_counter for the offline envs, which own the signed size directly.

    The live envs pass a direction read off the exchange (position_direction_from_status).
    Offline, position.position_size IS the truth, so the conversion belongs here rather
    than hand-rolled once per engine -- which is the drift this rule exists to stop.
    """
    size = position.position_size
    advance_hold_counter(position, math.copysign(1.0, size) if size else 0.0)


def advance_hold_counter(position: PositionState, direction: float) -> None:
    """Age the CURRENT position by one step, writing the new age to position.hold_counter.

        flat  -> long   a new position     -> 1
        long  -> long   the same position  -> 2, 3, 4, ...
        long  -> SHORT  a NEW position     -> 1     <-- a DIRECT flip, no flat bar between
        short -> flat   no position        -> 0

    The five live envs that track holding_time each hand-rolled this as "increment while a
    position exists, reset when flat". A
    direct flip never passes through flat, so the reset never fired and a one-step-old short
    reported the dead long's age as account_state[3] -- which the policy conditions on.
    """
    if direction == 0:
        position.hold_counter = 0
    elif direction != position.hold_direction:
        position.hold_counter = 1
    else:
        position.hold_counter += 1

    position.hold_direction = direction
