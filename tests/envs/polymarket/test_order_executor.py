"""Tests for Polymarket order executor."""

import pytest
from unittest.mock import MagicMock, patch

from tests.envs.polymarket.mocks import MockClobClient


def _make_order_args(token_id, amount, side):
    """Create a simple namespace object mimicking MarketOrderArgs."""
    obj = MagicMock()
    obj.token_id = token_id
    obj.amount = amount
    obj.side = side
    return obj


def _fok_sentinel():
    return "FOK"


class TestPolymarketOrderExecutor:
    """Tests for PolymarketOrderExecutor."""

    @pytest.fixture
    def mock_clob(self):
        return MockClobClient(initial_balance=10_000.0, yes_price=0.72)

    @pytest.fixture
    def executor(self, mock_clob):
        mock_order_type = MagicMock()
        mock_order_type.FOK = "FOK"
        with (
            patch(
                "torchtrade.envs.live.polymarket.order_executor.ClobClient",
                return_value=mock_clob,
            ),
            patch(
                "torchtrade.envs.live.polymarket.order_executor.MarketOrderArgs",
                side_effect=_make_order_args,
            ),
            patch(
                "torchtrade.envs.live.polymarket.order_executor.OrderType",
                mock_order_type,
            ),
        ):
            from torchtrade.envs.live.polymarket.order_executor import (
                PolymarketOrderExecutor,
            )

            exe = PolymarketOrderExecutor(private_key="0xtest", chain_id=137)
            exe.client = mock_clob
            yield exe

    def test_get_balance(self, executor):
        """get_balance returns USDC balance as float dollars."""
        balance = executor.get_balance()
        assert isinstance(balance, float)
        assert balance == pytest.approx(10_000.0)

    @pytest.mark.parametrize(
        "side,token_attr",
        [("yes", "yes_token_id"), ("no", "no_token_id")],
        ids=["buy-yes", "buy-no"],
    )
    def test_buy_shares(self, executor, side, token_attr):
        """buy() places a market order for YES or NO shares."""
        token_id = getattr(executor, token_attr, "tok_yes_1")
        result = executor.buy(token_id=token_id, amount_usdc=100.0)
        assert result["success"] is True

    def test_sell_shares(self, executor):
        """sell() sells shares from an existing position."""
        executor.buy(token_id="tok_yes_1", amount_usdc=100.0)
        shares = 50.0
        result = executor.sell(token_id="tok_yes_1", amount_shares=shares)
        assert result["success"] is True

    def test_get_positions(self, executor):
        """get_positions() returns current share holdings."""
        executor.buy(token_id="tok_yes_1", amount_usdc=100.0)
        positions = executor.get_positions()
        assert len(positions) > 0

    def test_cancel_all(self, executor):
        """cancel_all() returns True on success."""
        assert executor.cancel_all() is True

    def test_dry_run_mode(self, mock_clob):
        """In dry_run mode, trades are logged but not executed."""
        mock_order_type = MagicMock()
        mock_order_type.FOK = "FOK"
        with (
            patch(
                "torchtrade.envs.live.polymarket.order_executor.ClobClient",
                return_value=mock_clob,
            ),
            patch(
                "torchtrade.envs.live.polymarket.order_executor.MarketOrderArgs",
                side_effect=_make_order_args,
            ),
            patch(
                "torchtrade.envs.live.polymarket.order_executor.OrderType",
                mock_order_type,
            ),
        ):
            from torchtrade.envs.live.polymarket.order_executor import (
                PolymarketOrderExecutor,
            )

            exe = PolymarketOrderExecutor(
                private_key="0xtest", chain_id=137, dry_run=True
            )
            exe.client = mock_clob
            initial_balance = exe.get_balance()
            result = exe.buy(token_id="tok_yes_1", amount_usdc=100.0)
            assert result["success"] is True
            assert result.get("dry_run") is True
            assert exe.get_balance() == pytest.approx(initial_balance)
