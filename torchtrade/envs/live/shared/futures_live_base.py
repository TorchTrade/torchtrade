"""Shared base class for live futures trading environments.

Post-#253, the `_get_observation` bodies of all four futures exchanges (Binance, Bitget,
Bybit, OKX) are functionally identical -- the only differences were two dead locals
(`cash`, `entry_price`) in binance/bitget and cosmetic comments. This class holds the one
shared implementation so a future account_state fix only needs to land once.

Alpaca (spot) is NOT a futures env: it hardcodes leverage=1 and distance_to_liquidation=1.0
and reads cash rather than total_wallet_balance. It keeps its own `_get_observation` and
inherits `TorchTradeLiveEnv` directly.
"""
import torch
from tensordict import TensorDict, TensorDictBase

from torchtrade.envs.core.live import TorchTradeLiveEnv
from torchtrade.envs.core.state import advance_hold_counter, position_direction_from_status


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
        total_balance = balance.get("total_margin_balance", 0)
        position_status = status.get("position_status", None)

        # Dust is not a position: gating on `is None` let a 1e-12 residual left behind a
        # close take the position branch and read stale fields off it.
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
            position_size = position_status.qty
            position_value = abs(position_status.notional_value)
            current_price = position_status.mark_price
            unrealized_pnl_pct = position_status.unrealized_pnl_pct
            leverage = float(position_status.leverage)
            liquidation_price = position_status.liquidation_price

        # Build 6-element account state
        exposure_pct = position_value / total_balance if total_balance > 0 else 0.0

        if position_size == 0 or current_price == 0 or liquidation_price <= 0:
            distance_to_liquidation = 1.0
        else:
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
        return balance.get("total_margin_balance", 0)
