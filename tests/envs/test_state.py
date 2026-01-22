"""Tests for state management dataclasses."""

import pytest
from torchtrade.envs.state import HistoryTracker, FuturesHistoryTracker


class TestHistoryTracker:
    """Test HistoryTracker dataclass."""

    def test_initialization(self):
        """Test HistoryTracker initializes with empty lists."""
        history = HistoryTracker()
        assert history.base_prices == []
        assert history.actions == []
        assert history.rewards == []
        assert history.portfolio_values == []
        assert len(history) == 0

    def test_record_step(self):
        """Test recording a single step."""
        history = HistoryTracker()
        history.record_step(
            price=50000.0,
            action=1.0,
            reward=0.05,
            portfolio_value=5000.0
        )

        assert len(history) == 1
        assert history.base_prices == [50000.0]
        assert history.actions == [1.0]
        assert history.rewards == [0.05]
        assert history.portfolio_values == [5000.0]

    def test_record_multiple_steps(self):
        """Test recording multiple steps."""
        history = HistoryTracker()

        # Record first step
        history.record_step(
            price=50000.0,
            action=1.0,
            reward=0.05,
            portfolio_value=5000.0
        )

        # Record second step
        history.record_step(
            price=51000.0,
            action=0.0,
            reward=0.02,
            portfolio_value=5100.0
        )

        # Record third step
        history.record_step(
            price=49000.0,
            action=-1.0,
            reward=-0.04,
            portfolio_value=4900.0
        )

        assert len(history) == 3
        assert history.base_prices == [50000.0, 51000.0, 49000.0]
        assert history.actions == [1.0, 0.0, -1.0]
        assert history.rewards == [0.05, 0.02, -0.04]
        assert history.portfolio_values == [5000.0, 5100.0, 4900.0]

    def test_reset(self):
        """Test resetting history."""
        history = HistoryTracker()

        # Record some steps
        history.record_step(50000.0, 1.0, 0.05, 5000.0)
        history.record_step(51000.0, 0.0, 0.02, 5100.0)
        assert len(history) == 2

        # Reset
        history.reset()

        assert len(history) == 0
        assert history.base_prices == []
        assert history.actions == []
        assert history.rewards == []
        assert history.portfolio_values == []

    def test_to_dict(self):
        """Test exporting history as dictionary."""
        history = HistoryTracker()

        # Record some steps
        history.record_step(50000.0, 1.0, 0.05, 5000.0)
        history.record_step(51000.0, 0.0, 0.02, 5100.0)

        result = history.to_dict()

        assert isinstance(result, dict)
        assert result == {
            'base_prices': [50000.0, 51000.0],
            'actions': [1.0, 0.0],
            'rewards': [0.05, 0.02],
            'portfolio_values': [5000.0, 5100.0]
        }

    def test_to_dict_empty(self):
        """Test exporting empty history as dictionary."""
        history = HistoryTracker()
        result = history.to_dict()

        assert result == {
            'base_prices': [],
            'actions': [],
            'rewards': [],
            'portfolio_values': []
        }

    def test_len(self):
        """Test __len__ method."""
        history = HistoryTracker()
        assert len(history) == 0

        history.record_step(50000.0, 1.0, 0.05, 5000.0)
        assert len(history) == 1

        history.record_step(51000.0, 0.0, 0.02, 5100.0)
        assert len(history) == 2

        history.reset()
        assert len(history) == 0


class TestFuturesHistoryTracker:
    """Test FuturesHistoryTracker dataclass."""

    def test_initialization(self):
        """Test FuturesHistoryTracker initializes with empty lists."""
        history = FuturesHistoryTracker()
        assert history.base_prices == []
        assert history.actions == []
        assert history.rewards == []
        assert history.portfolio_values == []
        assert history.positions == []
        assert len(history) == 0

    def test_record_step_with_position(self):
        """Test recording a step with position."""
        history = FuturesHistoryTracker()
        history.record_step(
            price=50000.0,
            action=1.0,
            reward=0.05,
            portfolio_value=5000.0,
            position=0.5
        )

        assert len(history) == 1
        assert history.base_prices == [50000.0]
        assert history.actions == [1.0]
        assert history.rewards == [0.05]
        assert history.portfolio_values == [5000.0]
        assert history.positions == [0.5]

    def test_record_step_with_zero_position(self):
        """Test recording a step with zero position (no position)."""
        history = FuturesHistoryTracker()
        history.record_step(
            price=50000.0,
            action=1.0,
            reward=0.05,
            portfolio_value=5000.0,
            position=0.0
        )

        assert len(history) == 1
        assert history.base_prices == [50000.0]
        assert history.actions == [1.0]
        assert history.rewards == [0.05]
        assert history.portfolio_values == [5000.0]
        assert history.positions == [0.0]  # Zero position recorded

    def test_record_multiple_steps_with_positions(self):
        """Test recording multiple steps with positions."""
        history = FuturesHistoryTracker()

        # Long position
        history.record_step(50000.0, 1.0, 0.05, 5000.0, position=0.5)

        # Hold position
        history.record_step(51000.0, 0.0, 0.02, 5100.0, position=0.5)

        # Short position
        history.record_step(49000.0, -1.0, -0.04, 4900.0, position=-0.3)

        assert len(history) == 3
        assert history.positions == [0.5, 0.5, -0.3]

    def test_reset(self):
        """Test resetting futures history."""
        history = FuturesHistoryTracker()

        # Record some steps
        history.record_step(50000.0, 1.0, 0.05, 5000.0, position=0.5)
        history.record_step(51000.0, 0.0, 0.02, 5100.0, position=0.5)
        assert len(history) == 2

        # Reset
        history.reset()

        assert len(history) == 0
        assert history.base_prices == []
        assert history.actions == []
        assert history.rewards == []
        assert history.portfolio_values == []
        assert history.positions == []

    def test_to_dict(self):
        """Test exporting futures history as dictionary."""
        history = FuturesHistoryTracker()

        # Record some steps
        history.record_step(50000.0, 1.0, 0.05, 5000.0, position=0.5)
        history.record_step(51000.0, 0.0, 0.02, 5100.0, position=0.5)
        history.record_step(49000.0, -1.0, -0.04, 4900.0, position=-0.3)

        result = history.to_dict()

        assert isinstance(result, dict)
        assert result == {
            'base_prices': [50000.0, 51000.0, 49000.0],
            'actions': [1.0, 0.0, -1.0],
            'rewards': [0.05, 0.02, -0.04],
            'portfolio_values': [5000.0, 5100.0, 4900.0],
            'positions': [0.5, 0.5, -0.3],
            'action_types': ['hold', 'hold', 'hold']
        }

    def test_to_dict_empty(self):
        """Test exporting empty futures history as dictionary."""
        history = FuturesHistoryTracker()
        result = history.to_dict()

        assert result == {
            'base_prices': [],
            'actions': [],
            'rewards': [],
            'portfolio_values': [],
            'positions': [],
            'action_types': []
        }

    def test_inheritance(self):
        """Test that FuturesHistoryTracker inherits from HistoryTracker."""
        history = FuturesHistoryTracker()
        assert isinstance(history, HistoryTracker)
        assert isinstance(history, FuturesHistoryTracker)

    def test_mixed_position_values(self):
        """Test recording steps with different position values (including zero)."""
        history = FuturesHistoryTracker()

        # Record with long position
        history.record_step(50000.0, 1.0, 0.05, 5000.0, position=0.5)
        assert len(history.positions) == 1

        # Record with no position (zero)
        history.record_step(51000.0, 0.0, 0.02, 5100.0, position=0.0)
        assert len(history.positions) == 2

        # Record with short position
        history.record_step(49000.0, -1.0, -0.04, 4900.0, position=-0.3)
        assert len(history.positions) == 3
        assert history.positions == [0.5, 0.0, -0.3]
