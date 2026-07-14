"""Tests for the paper-only Polymarket order executor.

The V1 live path and the ~10 tests that pinned its call shape were deleted together:
``dry_run=False`` is now refused by the constructor, so none of it was reachable.

What is left is deliberately small. The class is a stub, and a stub's constants
(``cancel_all() is True``, the dry-run marker dict) cannot be usefully asserted -- a test that
re-states a `return True` is a tautology, not coverage. Only the REFUSALS can fail, so only
the refusals are tested here.
"""

import pytest

from torchtrade.envs.live.polymarket.order_executor import PolymarketOrderExecutor


class TestLiveIsRefused:
    """Makes the paper-only contract true for the PUBLIC API, not just for the env.

    PolymarketOrderExecutor is exported from ``torchtrade.envs.live``. With the config boundary
    on PolymarketBetEnv as the only gate, a caller using the exported class directly could
    still build a live CLOB client from a real key and post an order -- against an archived V1
    API, with no redemption workflow -- while the env believed itself paper-only.
    """

    def test_live_mode_raises(self):
        with pytest.raises(NotImplementedError) as exc:
            PolymarketOrderExecutor(dry_run=False)
        # Pin the SUBSTANCE. Matching a contentless phrase would let the message be reduced to
        # "not supported yet" -- or back to "pip install py-clob-client" -- and still pass.
        assert "archived" in str(exc.value)
        assert "redeem" in str(exc.value)

    def test_a_real_private_key_is_refused_not_swallowed(self):
        """The executor must not pocket a key it can never use.

        It previously accepted private_key/chain_id/signature_type/funder and ignored all four
        -- so a caller handing the EXPORTED class a funded key was told nothing, while the env
        refused the very same argument. "Refused, not ignored" is self-refuting if the class
        the env wraps does the ignoring.
        """
        with pytest.raises(TypeError, match="paper-only"):
            PolymarketOrderExecutor(private_key="0xREAL_SECRET")

    def test_empty_string_key_is_refused_too(self):
        """`if private_key:` would wave this through -- and the old example passed exactly
        os.getenv("POLYGON_PRIVATE_KEY", ""), i.e. "" whenever the var was unset."""
        with pytest.raises(TypeError, match="paper-only"):
            PolymarketOrderExecutor(private_key="")

    def test_the_dead_live_params_are_gone(self):
        """chain_id/signature_type/funder fed the deleted CLOB client. They must not linger as
        accepted-and-ignored params -- that is the same silent-swallow bug as the key."""
        with pytest.raises(TypeError):
            PolymarketOrderExecutor(signature_type=2, funder="0xproxy")


class TestPaperExecution:
    def test_default_construction_is_paper(self):
        """The DEFAULT must be the safe mode: it used to default to dry_run=False, so a caller
        who omitted the kwarg got a live client. Flipping it back makes this RAISE.

        The construction not raising IS the assertion. Asserting on buy()'s literal return, or
        on ._dry_run, would be asserting a constant against itself -- the class has no live
        state left to observe.
        """
        PolymarketOrderExecutor()

    def test_does_not_depend_on_py_clob_client_at_all(self):
        """Paper trading must not need the archived package. The module no longer imports it,
        so this is structural now rather than a runtime dry_run branch -- and this test is what
        stops the import shim being quietly resurrected."""
        import inspect

        import torchtrade.envs.live.polymarket.order_executor as oe

        # Assert against the module SOURCE, not one symbol name: pinning only `ClobClient`
        # would miss MarketOrderArgs / OrderType / BUY / a bare `import py_clob_client`.
        src = inspect.getsource(oe)
        assert "py_clob_client" not in src.replace("py-clob-client", "")
