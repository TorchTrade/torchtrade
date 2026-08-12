"""Shared base class for live futures trading environments.

Post-#253, the `_get_observation` bodies of all four futures exchanges (Binance, Bitget,
Bybit, OKX) are functionally identical -- the only differences were two dead locals
(`cash`, `entry_price`) in binance/bitget and cosmetic comments. This class holds the one
shared implementation so a future account_state fix only needs to land once.

Alpaca (spot) is NOT a futures env: it hardcodes leverage=1 and distance_to_liquidation=1.0
and reads cash rather than total_wallet_balance. It keeps its own `_get_observation` and
inherits `TorchTradeLiveEnv` directly.
"""
import math

import torch
from tensordict import TensorDict, TensorDictBase

from torchtrade.envs.core.live import TorchTradeLiveEnv
from torchtrade.envs.core.state import (
    advance_hold_counter,
    position_direction_from_status,
    position_qty_from_status,
)
from torchtrade.envs.utils.liquidation import nearest_liquidation_price


class TorchTradeFuturesLiveEnv(TorchTradeLiveEnv):
    """Base class for live futures trading environments (Binance, Bitget, Bybit, OKX).

    Holds the single _get_observation (account_state assembly) and _get_portfolio_value
    (total_margin_balance) shared by all four futures exchanges, so an account_state fix
    lands once here instead of in four drifting copies.

    Standard account state (6 elements):
    [exposure_pct, position_direction, unrealized_pnl_pct,
     holding_time, leverage, distance_to_liquidation]

    Subclasses (per-exchange base envs) must still implement:
    - _init_trading_clients(): Provider-specific client initialization
    - _build_observation_specs(): Provider-specific spec construction
    - _execute_trade_if_needed(): Trade execution logic
    - _reset(): Provider-specific reset scaffolding
    """

    def _get_observation(self, advance_hold: bool = True) -> TensorDictBase:
        """Get the current observation state.

        Args:
            advance_hold: If True (the default, used by `_step()`), ages `hold_counter`
                by one bar using the direction observed in THIS method's single
                `get_status()` call -- holding_time and position_direction in the
                emitted account_state are always derived from the same snapshot.
                `_reset()` passes False so a reset can never itself count a bar.
        """
        obs_dict = self.observer.get_observations(
            return_base_ohlc=self.config.include_base_features
        )

        if self.config.include_base_features:
            base_features = obs_dict.get("base_features")

        market_data = [obs_dict[features_name] for features_name in self.observer.get_keys()]

        # Get account state from trader (single fetch: holding_time and
        # position_direction below MUST come from this same snapshot)
        status = self.trader.get_status()
        balance = self.trader.get_account_balance()

        # exposure_pct denominator: use total_margin_balance (equity incl. unrealized PnL),
        # NOT total_wallet_balance. The latter's meaning diverges across exchanges -- Binance's
        # excludes uPnL while Bitget/Bybit/OKX map equity to the same key -- which made Binance's
        # exposure_pct read differently for the same position. total_margin_balance is uniformly
        # equity across all four, so exposure_pct is comparable cross-exchange (and matches the
        # portfolio value _get_portfolio_value returns).
        # Indexed, not .get(..., 0): all four adapters build this key unconditionally, so a
        # missing one means the adapter broke. The default turned that into a silent
        # exposure_pct of 0.0 -- every position reading as flat, forever (#277).
        total_balance = balance["total_margin_balance"]
        position_status = status.get("position_status", None)

        # Dust is not a position: gating on `is None` let a 1e-12 residual left behind a
        # close take the position branch and read stale fields off it. An unknown status
        # raises here rather than taking the flat branch below, which would report a held
        # position as flat (invariant #3) -- fail-closed like get_account_balance() above,
        # which already raises rather than inventing a balance. #295 adds a stale-but-marked
        # observation so a blip need not end the episode.
        position_direction = float(position_direction_from_status(position_status))
        if advance_hold:
            advance_hold_counter(self.position, position_direction)
        holding_time = float(self.position.hold_counter)

        if position_direction == 0:
            position_size = 0.0
            position_value = 0.0
            current_price = self.trader.get_mark_price()
            unrealized_pnl_pct = 0.0
            leverage = float(self.config.leverage)
            liquidation_price = 0.0
        else:
            # Every guard below is a comparison, and NaN compares False to all of them --
            # a NaN liquidation price would skip the fallback, reach the arithmetic, and
            # clamp to a distance of 0.0, telling the policy a healthy position is AT
            # liquidation on one garbage tick. Checked once, here, rather than at each
            # comparison (#277).
            # Every venue number this branch reads, not just the ones feeding
            # distance_to_liquidation: a NaN notional_value is worse than a NaN
            # liquidation price, because exposure_pct = nan/equity puts NaN straight into
            # the observation tensor and from there into the policy network.
            for _name, _value in (
                ("qty", position_status.qty),
                ("mark_price", position_status.mark_price),
                ("leverage", position_status.leverage),
                ("liquidation_price", position_status.liquidation_price),
                ("notional_value", position_status.notional_value),
                ("unrealized_pnl_pct", position_status.unrealized_pnl_pct),
                ("entry_price", position_status.entry_price),
            ):
                if not math.isfinite(_value):
                    raise ValueError(
                        f"venue reported a non-finite {_name} ({_value}) for an open "
                        f"position; refusing to derive an account state from it"
                    )
            position_size = position_status.qty
            position_value = abs(position_status.notional_value)
            current_price = position_status.mark_price
            unrealized_pnl_pct = position_status.unrealized_pnl_pct
            leverage = float(position_status.leverage)
            liquidation_price = position_status.liquidation_price

        # Build 6-element account state
        # Equity gone with the position still on: there is no exposure_pct to report, the
        # ratio being unbounded, and the old `else 0.0` reported the one value that is
        # certainly wrong -- a flat-looking account that is actually holding an underwater
        # position (invariant #3). Fail closed, like the unknown-status path above. An
        # account that is merely empty still reports 0.0 exposure, which is true.
        # `not (x > 0)` rather than `x <= 0`: NaN compares False to everything, so `<= 0`
        # skips the raise and the ternary below then hands back 0.0 -- a held position
        # reading flat, the exact bug this raise exists to prevent.
        # Keyed on position_direction (which goes through the dust rule), NOT on
        # position_value: a venue that omits the notional would zero the second
        # conjunct and skip the raise on a position that is very much held.
        if not (total_balance > 0) and position_direction != 0:
            raise ValueError(
                f"Position worth {position_value} held against non-positive equity "
                f"({total_balance}). The venue is mid-liquidation or reporting "
                f"inconsistently; refusing to report this account as flat."
            )
        exposure_pct = position_value / total_balance if total_balance > 0 else 0.0

        if position_size == 0 or current_price == 0:
            distance_to_liquidation = 1.0
        elif liquidation_price <= 0 and leverage == 1:
            # No leverage, nothing to be liquidated by, and the offline env says the same
            # via its has_liquidation gate -- so this must NOT fall through to the
            # arithmetic below, where a short would compute (0 - price)/price = -1 and
            # clamp to 0.0, reporting an unlevered position as AT liquidation.
            #
            # `== 1`, not `<= 1`: leverage of 0, negative or fractional is venue nonsense,
            # and letting it take this branch hands a held position the maximally-safe
            # answer -- #277 one field over. Those fall through to the helper, which
            # raises. Offline refuses the same inputs in __post_init__.
            distance_to_liquidation = 1.0
        else:
            if liquidation_price <= 0:
                # Cross-margin venues omit the liquidation price (OKX sends liqPx="" for
                # cross) because the whole account backs the position, so there is no
                # per-position price to publish. bybit blanks liqPrice for unified/cross
                # too, but blanks `leverage` with it, so its adapter yields
                # POSITION_UNKNOWN and raises before reaching here -- OKX is the venue
                # that actually exercises this path.
                # Defaulting to 1.0 reported a 20x long one move away from liquidation as
                # exactly as safe as a flat spot account (#277).
                #
                # Estimated from BOTH the isolated geometry and the account's equity,
                # taking whichever is nearer. Isolated alone is not the conservative
                # choice it looks like: it only sees this position, so once losses
                # elsewhere have eaten the collateral, cross liquidates earlier than
                # isolated says and the estimate would overstate the distance -- the same
                # fail-open, reintroduced with extra steps.
                #
                # Still not a guaranteed bound: a second cross position's maintenance
                # requirement is invisible to both estimates and would move liquidation
                # nearer again (#344). That assumption -- this env owns the account -- is
                # the same one exposure_pct and the bankruptcy baseline already make.
                liquidation_price = nearest_liquidation_price(
                    position_size=position_size,
                    entry_price=position_status.entry_price,
                    mark_price=current_price,
                    equity=total_balance,
                    leverage=leverage,
                )
            if position_size > 0:
                distance_to_liquidation = (current_price - liquidation_price) / current_price
            else:
                distance_to_liquidation = (liquidation_price - current_price) / current_price
            distance_to_liquidation = max(0.0, distance_to_liquidation)

        account_state = torch.tensor(
            [
                exposure_pct,
                position_direction,
                unrealized_pnl_pct,
                holding_time,
                leverage,
                distance_to_liquidation,
            ],
            dtype=torch.float,
        )

        out_td = TensorDict({self.account_state_key: account_state}, batch_size=())
        for market_data_name, data in zip(self.market_data_keys, market_data):
            out_td.set(market_data_name, torch.from_numpy(data))

        if self.config.include_base_features and base_features is not None:
            out_td.set("base_features", torch.from_numpy(base_features))

        return out_td

    def _get_portfolio_value(self) -> float:
        """Calculate total portfolio value (includes unrealized PnL)."""
        balance = self.trader.get_account_balance()
        # Indexed, for the same reason as _get_observation above: this is the `current`
        # side of is_bankrupt(), where a default of 0 is instant false bankruptcy.
        return balance["total_margin_balance"]

    def _get_current_position_quantity(self) -> float:
        """The signed size the exchange holds, dust read as flat.

        One copy: binance, bitget and bybit each carried a byte-identical
        `position.qty if position is not None else 0.0`, which returned the residual
        rather than 0 and let `abs(current_qty) > 0` fire on a flat account (#283).
        """
        return position_qty_from_status(self.trader.get_status().get("position_status"))
