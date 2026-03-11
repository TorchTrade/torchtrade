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
        # None + already flat = stays flat (steady-state HOLD)
        (None, 0, 0, False),
        # Direction flip: was long, exchange shows short (external action)
        (-0.005, 1, -1, False),
        # Direction flip: was short, exchange shows long (external action)
        (0.005, -1, 1, False),
    ], ids=[
        "sl_tp_closes_long",
        "sl_tp_closes_short",
        "flat_stays_flat",
        "drift_fix_long",
        "drift_fix_short",
        "long_matches",
        "short_matches",
        "none_closes_long",
        "none_stays_flat",
        "direction_flip_long_to_short",
        "direction_flip_short_to_long",
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

        # SL/TP levels should be reset on closure or direction flip
        if expected_closed or (prev_pos != 0 and expected_pos != 0 and prev_pos != expected_pos):
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

