"""Alpaca gets the close-position escape hatches the futures exchanges have (#289)."""

import pytest

from tests.mocks.alpaca import MockObserver, MockTrader
from torchtrade.envs.live.alpaca.env import AlpacaTorchTradingEnv, AlpacaTradingEnvConfig
from torchtrade.envs.live.alpaca.env_sltp import (
    AlpacaSLTPTorchTradingEnv,
    AlpacaSLTPTradingEnvConfig,
)

VARIANTS = [
    pytest.param(AlpacaTorchTradingEnv, AlpacaTradingEnvConfig, id="plain"),
    pytest.param(AlpacaSLTPTorchTradingEnv, AlpacaSLTPTradingEnvConfig, id="sltp"),
]


class _CountingTrader(MockTrader):
    """Records the flatten/cancel calls in ORDER, since order is part of the contract."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.calls = []

    @property
    def close_all_calls(self):
        return self.calls.count("close_all_positions")

    @property
    def close_symbol_calls(self):
        return self.calls.count("close_position")

    def close_all_positions(self):
        self.calls.append("close_all_positions")
        return super().close_all_positions()

    def close_position(self, *args, **kwargs):
        self.calls.append("close_position")
        return super().close_position(*args, **kwargs)

    def cancel_open_orders(self, *args, **kwargs):
        self.calls.append("cancel_open_orders")
        return super().cancel_open_orders(*args, **kwargs)


def _build(Env, Cfg, **flags):
    trader = _CountingTrader(initial_cash=10000.0)
    env = Env(
        config=Cfg(symbol="BTC/USD", window_sizes=[10], **flags),
        observer=MockObserver(window_sizes=[10]),
        trader=trader,
    )
    env._wait_for_next_timestamp = lambda: None
    return env, trader


@pytest.mark.parametrize("Env,Cfg", VARIANTS)
@pytest.mark.parametrize("on_init,expected", [(True, 1), (False, 0)])
def test_close_position_on_init_is_honoured(Env, Cfg, on_init, expected):
    """It was hardcoded, so there was no way to keep a position across a restart --
    the four futures exchanges have had this switch all along."""
    _, trader = _build(Env, Cfg, close_position_on_init=on_init)
    assert trader.close_symbol_calls == expected, (
        "the init flatten must be SYMBOL-scoped too: this flag defaults to True, so an "
        "account-wide call means merely constructing an env flattens unrelated holdings"
    )
    assert trader.close_all_calls == 0, "init must not touch the whole account"


@pytest.mark.parametrize("Env,Cfg", VARIANTS)
@pytest.mark.parametrize("on_reset,expected_after_reset", [(True, 1), (False, 0)])
def test_close_position_on_reset_is_honoured(Env, Cfg, on_reset, expected_after_reset):
    """Alpaca's _reset cancelled orders but never closed, so a stale position survived
    every episode boundary with no way to opt into flattening it."""
    env, trader = _build(Env, Cfg, close_position_on_init=False,
                         close_position_on_reset=on_reset)
    assert trader.close_all_calls == 0, "init must not have flattened"

    env.reset()
    assert trader.close_symbol_calls == expected_after_reset, (
        "the reset flatten must be SYMBOL-scoped: close_all_positions() iterates the "
        "whole account, so an opt-in reset would flatten a second env's symbol or a "
        "manual holding at every episode boundary"
    )
    assert trader.close_all_calls == 0, "reset must not touch the whole account"


@pytest.mark.parametrize("Env,Cfg", VARIANTS)
def test_orders_are_cancelled_before_anything_is_closed(Env, Cfg):
    """Cancel first, at BOTH sites, as all four futures envs do.

    `cancel_open_orders()` falls through to the account-wide cancel-all, and
    `close_all_positions()` submits market closes without blocking -- so close-then-cancel
    cancelled the close it had just submitted. Outside RTH it could not fill first, and
    the flatten was silently reverted. No test pinned the ordering, so a mutation
    restoring it survived.
    """
    env, trader = _build(Env, Cfg, close_position_on_init=True,
                         close_position_on_reset=True)

    def _assert_cancel_first(calls, where):
        flattens = [i for i, c in enumerate(calls) if c.startswith("close_")]
        cancels = [i for i, c in enumerate(calls) if c == "cancel_open_orders"]
        assert cancels and flattens, f"expected both at {where}: {calls}"
        assert min(cancels) < min(flattens), (
            f"at {where} a close was submitted before the cancel-all that revokes it: "
            f"{calls}"
        )

    # BOTH sites. The first version of this test cleared the log before reset, so it
    # only ever saw the reset ordering -- and the init ordering was the broken one.
    _assert_cancel_first(list(trader.calls), "__init__")
    trader.calls.clear()
    env.reset()
    _assert_cancel_first(list(trader.calls), "_reset")


@pytest.mark.parametrize("Env,Cfg", VARIANTS)
def test_the_defaults_reproduce_the_previous_behaviour(Env, Cfg):
    """Flatten on construction, not on reset -- exactly what the hardcoded version did,
    so no existing config changes meaning."""
    config = Cfg(symbol="BTC/USD", window_sizes=[10])
    assert (config.close_position_on_init, config.close_position_on_reset) == (True, False)


@pytest.mark.parametrize("Env,Cfg", VARIANTS)
def test_close_does_not_flatten_the_account(Env, Cfg):
    """It did, unconditionally, which made `close_position_on_init=False` pointless --
    the position it preserved was market-closed the moment the process exited cleanly.
    The four futures envs only warn here. `examples/online_rl/` calls env.close()."""
    env, trader = _build(Env, Cfg, close_position_on_init=False)
    trader.calls.clear()

    env.close()

    assert trader.close_all_calls == 0 and trader.close_symbol_calls == 0, (
        f"close() flattened the account: {trader.calls}"
    )
    assert "cancel_open_orders" in trader.calls, "close() must still cancel orders"


@pytest.mark.parametrize("Env,Cfg", VARIANTS)
def test_a_flat_account_does_not_warn_that_the_reset_flatten_failed(Env, Cfg, caplog):
    """On alpaca, `close_position()` returns False for the state we WANTED.

    It wraps a client call that raises when the symbol is already flat (code 40410000),
    so a bare `if not closed` warns on every reset of a flat account -- training the
    operator to ignore the one warning that is real. bybit returns True when flat, which
    is why the copied pattern misfired. The check re-reads the status instead.
    """
    class _FlatTrader(_CountingTrader):
        def close_position(self, *args, **kwargs):
            super().close_position(*args, **kwargs)
            return False  # alpaca's wrapper does this when the symbol is already flat

    trader = _FlatTrader(initial_cash=10000.0)
    env = Env(
        config=Cfg(symbol="BTC/USD", window_sizes=[10],
                   close_position_on_init=False, close_position_on_reset=True),
        observer=MockObserver(window_sizes=[10]),
        trader=trader,
    )
    env._wait_for_next_timestamp = lambda: None

    with caplog.at_level("WARNING"):
        env.reset()

    assert not [r for r in caplog.records if "close_position_on_reset failed" in r.getMessage()], (
        "warned that the flatten failed on an account that is already flat"
    )
