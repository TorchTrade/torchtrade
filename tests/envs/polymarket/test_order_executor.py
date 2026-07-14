"""Tests for the paper-only Polymarket order executor.

The V1 live path (ClobClient construction, MarketOrderArgs/OrderType.FOK order posting) and
the ~10 tests that pinned its call shape were deleted together: ``dry_run=False`` is now
REFUSED by the constructor, so none of it was reachable. What remains is the contract that
actually matters -- this class cannot submit an order, and cannot be talked into trying.
"""

import pytest

from torchtrade.envs.live.polymarket.order_executor import PolymarketOrderExecutor


class TestLiveIsRefused:
    """Makes the paper-only contract true for the PUBLIC API, not just for the env.

    PolymarketOrderExecutor is exported from ``torchtrade.envs.live``. With the config
    boundary on PolymarketBetEnv as the only gate, a caller using the exported class directly
    could still build a live CLOB client from a real private key and post an order -- against
    an archived V1 API, with no redemption workflow. The env believing itself paper-only did
    nothing to stop that.
    """

    def test_live_mode_raises(self):
        with pytest.raises(NotImplementedError) as exc:
            PolymarketOrderExecutor(private_key="0xtest", dry_run=False)
        # Pin the SUBSTANCE, not a contentless phrase: both blockers must survive, or the
        # message could be reduced to "not supported yet" and still pass.
        assert "archived" in str(exc.value)
        assert "redeem" in str(exc.value)

    def test_live_mode_raises_even_with_a_funded_looking_config(self):
        """A full live config -- key, chain, proxy signature type, funder -- must not sneak
        past. The refusal is on dry_run, not on whether the caller looks prepared."""
        with pytest.raises(NotImplementedError):
            PolymarketOrderExecutor(
                private_key="0xdeadbeef",
                chain_id=137,
                signature_type=2,
                funder="0xproxy",
                dry_run=False,
            )


class TestPaperExecution:
    def test_defaults_to_paper(self):
        """The DEFAULT must be the safe mode. It used to default to dry_run=False, so any
        caller who forgot the kwarg -- including the env -- got a live client."""
        assert PolymarketOrderExecutor(private_key="x")._dry_run is True

    def test_no_clob_client_is_ever_constructed(self):
        """Constructing a ClobClient would derive API creds -- a network roundtrip against the
        wallet. That shipped as a real bug once; it is now structurally impossible."""
        assert PolymarketOrderExecutor(private_key="0xtest").client is None

    def test_buy_submits_nothing(self):
        assert PolymarketOrderExecutor().buy(token_id="tok", amount_usdc=10.0) == {
            "success": True,
            "dry_run": True,
        }

    def test_cancel_all_is_a_noop(self):
        assert PolymarketOrderExecutor().cancel_all() is True

    def test_does_not_depend_on_py_clob_client_at_all(self):
        """Paper trading must not need the archived package. The module no longer imports it,
        so this is structural now rather than a runtime dry_run branch."""
        import torchtrade.envs.live.polymarket.order_executor as oe

        assert not hasattr(oe, "ClobClient")
