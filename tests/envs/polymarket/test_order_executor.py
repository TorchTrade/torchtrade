"""Tests for Polymarket order executor."""

import pytest
from unittest.mock import MagicMock, patch

from tests.envs.polymarket.mocks import MockClobClient
from torchtrade.envs.live.polymarket.order_executor import PolymarketOrderExecutor


def _make_order_args(token_id, amount, side):
    obj = MagicMock()
    obj.token_id = token_id
    obj.amount = amount
    obj.side = side
    return obj


@pytest.fixture
def mock_clob():
    return MockClobClient(initial_balance=10_000.0, yes_price=0.72)


@pytest.fixture
def patched_module():
    """Keep MarketOrderArgs/OrderType/ClobClient patched for the whole test."""
    mock_order_type = MagicMock()
    mock_order_type.FOK = "FOK"
    patches = [
        patch(
            "torchtrade.envs.live.polymarket.order_executor.ClobClient",
            new=MagicMock(),
        ),
        patch(
            "torchtrade.envs.live.polymarket.order_executor.MarketOrderArgs",
            side_effect=_make_order_args,
        ),
        patch(
            "torchtrade.envs.live.polymarket.order_executor.OrderType",
            new=mock_order_type,
        ),
    ]
    for p in patches:
        p.start()
    yield
    for p in patches:
        p.stop()


@pytest.fixture
def executor(mock_clob, patched_module):
    exe = PolymarketOrderExecutor(private_key="0xtest", chain_id=137)
    exe.client = mock_clob
    return exe


class TestPolymarketOrderExecutor:
    """Tests for PolymarketOrderExecutor."""

    def test_get_balance(self, executor):
        """get_balance returns USDC balance as float dollars."""
        assert executor.get_balance() == pytest.approx(10_000.0)

    @pytest.mark.parametrize(
        "token_id",
        ["tok_yes_1", "tok_no_1"],
        ids=["buy-yes", "buy-no"],
    )
    def test_buy_shares(self, executor, token_id):
        """buy() places a market order for YES or NO shares."""
        assert executor.buy(token_id=token_id, amount_usdc=100.0)["success"] is True

    def test_sell_shares(self, executor):
        """sell() sells shares from an existing position."""
        executor.buy(token_id="tok_yes_1", amount_usdc=100.0)
        assert executor.sell(token_id="tok_yes_1", amount_shares=50.0)["success"] is True

    def test_get_positions(self, executor):
        """get_positions() returns current share holdings after a buy."""
        executor.buy(token_id="tok_yes_1", amount_usdc=100.0)
        assert len(executor.get_positions()) > 0

    def test_cancel_all(self, executor):
        """cancel_all() returns True on success."""
        assert executor.cancel_all() is True

    def test_dry_run_mode_does_not_touch_balance(self, mock_clob, patched_module):
        """dry_run=True returns success without altering wallet balance."""
        exe = PolymarketOrderExecutor(private_key="0xtest", chain_id=137, dry_run=True)
        exe.client = mock_clob
        initial = exe.get_balance()
        result = exe.buy(token_id="tok_yes_1", amount_usdc=100.0)
        assert result["success"] is True
        assert result.get("dry_run") is True
        assert exe.get_balance() == pytest.approx(initial)

    @pytest.mark.parametrize(
        "method,args",
        [
            ("buy", {"token_id": "tok", "amount_usdc": 1.0}),
            ("sell", {"token_id": "tok", "amount_shares": 1.0}),
        ],
        ids=["buy-error", "sell-error"],
    )
    def test_place_market_returns_failure_dict_on_exception(
        self, executor, method, args
    ):
        """post_order failures surface as success=False — env relies on this contract."""
        executor.client.post_order = MagicMock(side_effect=Exception("api down"))
        result = getattr(executor, method)(**args)
        assert result == {"success": False, "error": "api down"}

    def test_close_position_calls_sell_when_size_positive(self, executor):
        """close_position sells the held shares for the matching token."""
        executor.buy(token_id="tok_yes_1", amount_usdc=72.0)
        result = executor.close_position("tok_yes_1")
        assert result["success"] is True

    def test_close_position_already_flat_when_no_match(self, executor):
        """close_position short-circuits when no matching position exists."""
        result = executor.close_position("tok_unknown")
        assert result == {"success": True, "already_flat": True}

    def test_close_position_returns_error_on_failure(self, executor):
        """get_positions failure inside close_position surfaces as success=False."""
        executor.client.get_positions = MagicMock(side_effect=Exception("rpc down"))
        result = executor.close_position("tok_yes_1")
        assert result["success"] is False
        assert "rpc down" in result["error"]

    def test_get_balance_returns_zero_on_api_failure(self, executor):
        """API failure inside get_balance returns 0.0 rather than raising."""
        executor.client.get_balance_allowance = MagicMock(side_effect=Exception("503"))
        assert executor.get_balance() == 0.0

    def test_get_positions_returns_empty_on_api_failure(self, executor):
        """API failure inside get_positions returns []."""
        executor.client.get_positions = MagicMock(side_effect=Exception("503"))
        assert executor.get_positions() == []

    def test_cancel_all_returns_false_on_api_failure(self, executor):
        """API failure inside cancel_all returns False."""
        executor.client.cancel_all = MagicMock(side_effect=Exception("503"))
        assert executor.cancel_all() is False


class TestDryRunWithoutClient:
    """dry_run=True must work even when py-clob-client isn't installed."""

    def test_constructs_without_clob_dependency(self, monkeypatch):
        monkeypatch.setattr(
            "torchtrade.envs.live.polymarket.order_executor.ClobClient", None
        )
        exe = PolymarketOrderExecutor(private_key="0x", dry_run=True)
        assert exe.client is None

    def test_methods_return_safe_defaults_without_client(self, monkeypatch):
        monkeypatch.setattr(
            "torchtrade.envs.live.polymarket.order_executor.ClobClient", None
        )
        exe = PolymarketOrderExecutor(private_key="0x", dry_run=True)
        assert exe.get_balance() == 0.0
        assert exe.get_positions() == []
        assert exe.buy("tok", 1.0)["success"] is True
        assert exe.sell("tok", 1.0)["success"] is True
        assert exe.close_position("tok") == {"success": True, "already_flat": True}

    def test_live_mode_still_requires_clob(self, monkeypatch):
        monkeypatch.setattr(
            "torchtrade.envs.live.polymarket.order_executor.ClobClient", None
        )
        with pytest.raises(ImportError, match="py-clob-client is required"):
            PolymarketOrderExecutor(private_key="0x", dry_run=False)
