"""Tests for state management dataclasses."""

from torchtrade.envs.core.state import HistoryTracker


class TestHistoryTracker:
    """Test HistoryTracker dataclass."""

    def test_initialization(self):
        """Test HistoryTracker initializes with empty lists."""
        history = HistoryTracker()
        assert history.base_prices == []
        assert history.actions == []
        assert history.rewards == []
        assert history.portfolio_values == []
        assert history.positions == []
        assert history.action_types == []
        assert len(history) == 0

    def test_record_step_without_position(self):
        """Test recording a step without specifying position (defaults to 0.0)."""
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
        assert history.positions == [0.0]  # Default position
        assert history.action_types == ['hold']

    def test_record_step_with_position(self):
        """Test recording a step with explicit position."""
        history = HistoryTracker()
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

    def test_record_multiple_steps(self):
        """Test recording multiple steps with positions."""
        history = HistoryTracker()

        # Long position
        history.record_step(50000.0, 1.0, 0.05, 5000.0, position=0.5)

        # Hold position
        history.record_step(51000.0, 0.0, 0.02, 5100.0, position=0.5)

        # Short position
        history.record_step(49000.0, -1.0, -0.04, 4900.0, position=-0.3)

        assert len(history) == 3
        assert history.base_prices == [50000.0, 51000.0, 49000.0]
        assert history.actions == [1.0, 0.0, -1.0]
        assert history.rewards == [0.05, 0.02, -0.04]
        assert history.portfolio_values == [5000.0, 5100.0, 4900.0]
        assert history.positions == [0.5, 0.5, -0.3]
        assert history.action_types == ['hold', 'hold', 'hold']

    def test_reset(self):
        """Test resetting history."""
        history = HistoryTracker()

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
        assert history.action_types == []

    def test_to_dict(self):
        """Test exporting history as dictionary."""
        history = HistoryTracker()

        # Record some steps
        history.record_step(50000.0, 1.0, 0.05, 5000.0, position=0.5)
        history.record_step(51000.0, 0.0, 0.02, 5100.0, position=0.5)

        result = history.to_dict()

        assert isinstance(result, dict)
        assert result == {
            'base_prices': [50000.0, 51000.0],
            'actions': [1.0, 0.0],
            'rewards': [0.05, 0.02],
            'portfolio_values': [5000.0, 5100.0],
            'positions': [0.5, 0.5],
            'action_types': ['hold', 'hold']
        }

    def test_to_dict_empty(self):
        """Test exporting empty history as dictionary."""
        history = HistoryTracker()
        result = history.to_dict()

        assert result == {
            'base_prices': [],
            'actions': [],
            'rewards': [],
            'portfolio_values': [],
            'positions': [],
            'action_types': []
        }

    def test_len(self):
        """Test __len__ method."""
        history = HistoryTracker()
        assert len(history) == 0

        history.record_step(50000.0, 1.0, 0.05, 5000.0, position=0.5)
        assert len(history) == 1

        history.record_step(51000.0, 0.0, 0.02, 5100.0, position=0.5)
        assert len(history) == 2

        history.reset()
        assert len(history) == 0

    def test_mixed_position_values(self):
        """Test recording steps with different position values (including zero)."""
        history = HistoryTracker()

        # Long position
        history.record_step(50000.0, 1.0, 0.05, 5000.0, position=0.5)
        assert len(history.positions) == 1

        # No position (zero)
        history.record_step(51000.0, 0.0, 0.02, 5100.0, position=0.0)
        assert len(history.positions) == 2

        # Short position
        history.record_step(49000.0, -1.0, -0.04, 4900.0, position=-0.3)
        assert len(history.positions) == 3
        assert history.positions == [0.5, 0.0, -0.3]
