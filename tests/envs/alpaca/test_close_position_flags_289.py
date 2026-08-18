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
    """Counts the flatten calls, since that is the behaviour under test."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.close_all_calls = 0

    def close_all_positions(self):
        self.close_all_calls += 1
        return super().close_all_positions()


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
    assert trader.close_all_calls == expected


@pytest.mark.parametrize("Env,Cfg", VARIANTS)
@pytest.mark.parametrize("on_reset,expected_after_reset", [(True, 1), (False, 0)])
def test_close_position_on_reset_is_honoured(Env, Cfg, on_reset, expected_after_reset):
    """Alpaca's _reset cancelled orders but never closed, so a stale position survived
    every episode boundary with no way to opt into flattening it."""
    env, trader = _build(Env, Cfg, close_position_on_init=False,
                         close_position_on_reset=on_reset)
    assert trader.close_all_calls == 0, "init must not have flattened"

    env.reset()
    assert trader.close_all_calls == expected_after_reset


@pytest.mark.parametrize("Env,Cfg", VARIANTS)
def test_the_defaults_reproduce_the_previous_behaviour(Env, Cfg):
    """Flatten on construction, not on reset -- exactly what the hardcoded version did,
    so no existing config changes meaning."""
    config = Cfg(symbol="BTC/USD", window_sizes=[10])
    assert (config.close_position_on_init, config.close_position_on_reset) == (True, False)
