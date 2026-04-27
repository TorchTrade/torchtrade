"""Tests for the slim Polymarket order executor (buy + cancel_all)."""

from unittest.mock import MagicMock, patch

import pytest

from torchtrade.envs.live.polymarket.order_executor import PolymarketOrderExecutor


def _make_order_args(token_id, amount, side):
    obj = MagicMock()
    obj.token_id = token_id
    obj.amount = amount
    obj.side = side
    return obj


@pytest.fixture
def patched_module():
    """Patch the optional py-clob-client imports for the whole test."""
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
def executor(patched_module):
    exe = PolymarketOrderExecutor(private_key="0xtest", chain_id=137)
    exe.client = MagicMock()
    exe.client.post_order.return_value = {"success": True, "orderID": "ord-1"}
    return exe


class TestBuy:
    def test_returns_success_dict(self, executor):
        result = executor.buy(token_id="tok_yes_1", amount_usdc=100.0)
        assert result["success"] is True

    def test_returns_failure_dict_on_post_order_exception(self, executor):
        executor.client.post_order = MagicMock(side_effect=Exception("api down"))
        result = executor.buy(token_id="tok", amount_usdc=1.0)
        assert result == {"success": False, "error": "api down"}

    def test_pins_buy_side_and_fok_order_type(self, executor):
        """A BUY↔SELL or FOK↔GTC swap must fail this test, not silently pass."""
        from torchtrade.envs.live.polymarket import order_executor as oe

        executor.buy(token_id="tok_yes", amount_usdc=42.0)
        # MarketOrderArgs called with side=BUY (the constant from the package)
        args_call = oe.MarketOrderArgs.call_args
        assert args_call.kwargs["side"] == oe.BUY
        assert args_call.kwargs["token_id"] == "tok_yes"
        assert args_call.kwargs["amount"] == 42.0
        # post_order called with OrderType.FOK (fill-or-kill)
        post_call = executor.client.post_order.call_args
        assert post_call.args[1] == oe.OrderType.FOK


class TestCancelAll:
    def test_returns_true_on_success(self, executor):
        assert executor.cancel_all() is True

    def test_returns_false_on_api_failure(self, executor):
        executor.client.cancel_all = MagicMock(side_effect=Exception("503"))
        assert executor.cancel_all() is False


class TestDryRunWithoutClient:
    """``dry_run=True`` works without py-clob-client installed."""

    def test_constructs_with_no_clob_dependency(self, monkeypatch):
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
        assert exe.buy("tok", 1.0)["success"] is True
        assert exe.cancel_all() is True

    def test_live_mode_still_requires_clob(self, monkeypatch):
        monkeypatch.setattr(
            "torchtrade.envs.live.polymarket.order_executor.ClobClient", None
        )
        with pytest.raises(ImportError, match="py-clob-client is required"):
            PolymarketOrderExecutor(private_key="0x", dry_run=False)


class TestDryRunSkipsRealOrder:
    def test_buy_in_dry_run_returns_dry_run_marker(self, patched_module):
        exe = PolymarketOrderExecutor(private_key="0x", dry_run=True)
        exe.client = MagicMock()  # would error if hit
        result = exe.buy(token_id="tok", amount_usdc=10.0)
        assert result == {"success": True, "dry_run": True}
        exe.client.create_market_order.assert_not_called()
        exe.client.post_order.assert_not_called()


class TestDryRunSkipsClientConstructionEvenWhenAvailable:
    """dry_run=True must not touch the network even when py-clob-client IS installed.

    The previous behavior constructed a real ``ClobClient`` and called
    ``set_api_creds(create_or_derive_api_creds())`` (a network roundtrip + valid
    private key requirement) whenever the package was available — regardless of
    ``dry_run``. Paper trading should be fully offline.
    """

    def test_dry_run_does_not_construct_clob_client(self):
        mock_clob_class = MagicMock()
        with patch(
            "torchtrade.envs.live.polymarket.order_executor.ClobClient",
            new=mock_clob_class,
        ):
            exe = PolymarketOrderExecutor(private_key="0xtest", dry_run=True)
        assert exe.client is None
        mock_clob_class.assert_not_called()

    def test_dry_run_does_not_derive_api_creds(self):
        """Critical: the API-creds derivation hits the wallet RPC."""
        mock_clob_instance = MagicMock()
        mock_clob_class = MagicMock(return_value=mock_clob_instance)
        with patch(
            "torchtrade.envs.live.polymarket.order_executor.ClobClient",
            new=mock_clob_class,
        ):
            PolymarketOrderExecutor(private_key="0xtest", dry_run=True)
        mock_clob_instance.create_or_derive_api_creds.assert_not_called()
        mock_clob_instance.set_api_creds.assert_not_called()

    def test_dry_run_buy_returns_marker_when_package_available(self):
        """End-to-end dry-run contract: even with py-clob-client installed,
        ``buy()`` must return the dry-run marker without touching the network."""
        mock_clob_class = MagicMock()
        with patch(
            "torchtrade.envs.live.polymarket.order_executor.ClobClient",
            new=mock_clob_class,
        ):
            exe = PolymarketOrderExecutor(private_key="0xtest", dry_run=True)
        result = exe.buy(token_id="tok", amount_usdc=10.0)
        assert result == {"success": True, "dry_run": True}
        mock_clob_class.assert_not_called()
