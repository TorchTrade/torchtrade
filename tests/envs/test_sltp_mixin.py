"""Tests for SLTPMixin position sync logic."""

import pytest
from dataclasses import dataclass
from torchtrade.envs.utils.sltp_mixin import SLTPMixin
from torchtrade.envs.core.state import PositionState


@dataclass
class FakePositionStatus:
    """Minimal position status for testing."""
    qty: float


class FakeSLTPEnv(SLTPMixin):
    """Minimal environment using SLTPMixin for testing."""

    def __init__(self):
        self.position = PositionState()
        self.active_stop_loss = 0.0
        self.active_take_profit = 0.0


class TestSyncPositionFromExchange:
    """Tests for _sync_position_from_exchange — the core anti-stacking fix."""

    @pytest.fixture
    def env(self):
        return FakeSLTPEnv()

    @pytest.mark.parametrize("exchange_qty,prev_pos,expected_pos,expected_closed", [
        # SL/TP closure: had long, exchange is flat
        (0.0, 1, 0, True),
        # SL/TP closure: had short, exchange is flat
        (0.0, -1, 0, True),
        # No closure: already flat, exchange flat
        (0.0, 0, 0, False),
        # Drift fix: env thinks flat, exchange has long (failed bracket scenario)
        (0.005, 0, 1, False),
        # Drift fix: env thinks flat, exchange has short
        (-0.005, 0, -1, False),
        # No change: long position matches
        (0.005, 1, 1, False),
        # No change: short position matches
        (-0.005, -1, -1, False),
        # Position status is None (no position on exchange)
        (None, 1, 0, True),
    ], ids=[
        "sl_tp_closes_long",
        "sl_tp_closes_short",
        "flat_stays_flat",
        "drift_fix_long",
        "drift_fix_short",
        "long_matches",
        "short_matches",
        "none_closes_long",
    ])
    def test_position_sync(self, env, exchange_qty, prev_pos, expected_pos, expected_closed):
        """Position state must sync from exchange, detecting closures and fixing drift."""
        env.position.current_position = prev_pos
        env.active_stop_loss = 48000.0
        env.active_take_profit = 52000.0

        if exchange_qty is None:
            position_status = None
        else:
            position_status = FakePositionStatus(qty=exchange_qty)

        closed = env._sync_position_from_exchange(position_status)

        assert closed == expected_closed
        assert env.position.current_position == expected_pos

        # SL/TP levels should be reset on closure
        if expected_closed:
            assert env.active_stop_loss == 0.0
            assert env.active_take_profit == 0.0

    def test_sltp_levels_preserved_when_no_closure(self, env):
        """Active SL/TP levels should NOT be reset when position persists."""
        env.position.current_position = 1
        env.active_stop_loss = 48000.0
        env.active_take_profit = 52000.0

        env._sync_position_from_exchange(FakePositionStatus(qty=0.005))

        assert env.active_stop_loss == 48000.0
        assert env.active_take_profit == 52000.0

    def test_stacking_prevention_scenario(self, env):
        """Reproduce the stacking bug: main order succeeds, SL/TP fails, state drifts.

        Without the sync fix, the env would think it's flat (current_position=0)
        while the exchange has an open position, allowing duplicate orders.
        """
        # Step 1: Trade executed, main order succeeded but SL/TP failed.
        # The env never updated current_position (it stayed 0).
        env.position.current_position = 0  # Bug state: env thinks flat

        # Step 2: Exchange actually has a long position
        exchange_status = FakePositionStatus(qty=0.005)

        # Step 3: _sync_position_from_exchange corrects the drift
        closed = env._sync_position_from_exchange(exchange_status)

        assert closed is False  # Not a closure
        assert env.position.current_position == 1  # Now correctly reflects exchange

        # Step 4: Duplicate guard would now correctly block re-entry
        # (tested in env_sltp tests, but the precondition is the sync above)
