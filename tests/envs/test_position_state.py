"""Tests for PositionState dataclass."""

from torchtrade.envs.core.state import PositionState


class TestPositionStateInitialization:
    """Tests for PositionState initialization."""

    def test_default_initialization(self):
        """PositionState should initialize with zero values."""
        position = PositionState()

        assert position.current_position == 0.0
        assert position.position_size == 0.0
        assert position.position_value == 0.0
        assert position.entry_price == 0.0
        assert position.unrealized_pnlpc == 0.0
        assert position.hold_counter == 0

    def test_custom_initialization(self):
        """PositionState should accept custom initialization values."""
        position = PositionState(
            current_position=1.0,
            position_size=10.5,
            position_value=1050.0,
            entry_price=100.0,
            unrealized_pnlpc=0.05,
            hold_counter=5
        )

        assert position.current_position == 1.0
        assert position.position_size == 10.5
        assert position.position_value == 1050.0
        assert position.entry_price == 100.0
        assert position.unrealized_pnlpc == 0.05
        assert position.hold_counter == 5


class TestPositionStateReset:
    """Tests for PositionState reset method."""

    def test_reset_clears_all_values(self):
        """Reset should clear all position state to initial values."""
        position = PositionState(
            current_position=1.0,
            position_size=10.5,
            position_value=1050.0,
            entry_price=100.0,
            unrealized_pnlpc=0.05,
            hold_counter=5
        )

        position.reset()

        assert position.current_position == 0.0
        assert position.position_size == 0.0
        assert position.position_value == 0.0
        assert position.entry_price == 0.0
        assert position.unrealized_pnlpc == 0.0
        assert position.hold_counter == 0

    def test_reset_multiple_times(self):
        """Reset should work correctly when called multiple times."""
        position = PositionState()

        # Set some values
        position.position_size = 5.0
        position.hold_counter = 10
        position.reset()

        assert position.position_size == 0.0
        assert position.hold_counter == 0

        # Set values again
        position.entry_price = 200.0
        position.unrealized_pnlpc = 0.1
        position.reset()

        assert position.entry_price == 0.0
        assert position.unrealized_pnlpc == 0.0


class TestPositionStateAttributes:
    """Tests for PositionState attribute manipulation."""

    def test_attribute_modification(self):
        """Attributes should be modifiable."""
        position = PositionState()

        position.current_position = 1.0
        assert position.current_position == 1.0

        position.position_size = 15.5
        assert position.position_size == 15.5

        position.hold_counter = 3
        assert position.hold_counter == 3

    def test_negative_position_size(self):
        """Position size should accept negative values (for shorts)."""
        position = PositionState(position_size=-10.0)
        assert position.position_size == -10.0

    def test_negative_unrealized_pnl(self):
        """Unrealized PnL can be negative (losses)."""
        position = PositionState(unrealized_pnlpc=-0.05)
        assert position.unrealized_pnlpc == -0.05


class TestPositionStateIntegration:
    """Integration tests for PositionState usage patterns."""

    def test_position_lifecycle(self):
        """Test a typical position lifecycle."""
        position = PositionState()

        # Open position
        position.current_position = 1.0
        position.position_size = 10.0
        position.entry_price = 100.0
        position.position_value = 1000.0
        position.hold_counter = 0

        # Hold for a few steps
        position.hold_counter = 1
        position.hold_counter = 2
        position.hold_counter = 3

        # Update unrealized PnL
        position.unrealized_pnlpc = 0.05  # 5% profit

        # Close position
        position.reset()

        assert position.current_position == 0.0
        assert position.position_size == 0.0
        assert position.hold_counter == 0

    def test_short_position_pattern(self):
        """Test pattern for short positions."""
        position = PositionState()

        # Open short position
        position.current_position = -1.0
        position.position_size = -10.0
        position.entry_price = 100.0

        assert position.current_position == -1.0
        assert position.position_size < 0

        # Profit on short (price goes down)
        position.unrealized_pnlpc = 0.1  # 10% profit

        assert position.unrealized_pnlpc > 0
