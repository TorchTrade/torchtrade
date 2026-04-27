"""Tests for PolyTimeBarEnv."""

import logging
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from tensordict import TensorDict
from torchrl.data import Bounded

from tests.envs.polymarket.mocks import MockPolymarketObserver, MockPolymarketTrader
from torchtrade.envs.live.polymarket.env import (
    PolyTimeBarEnv,
    PolyTimeBarEnvConfig,
    _parse_execute_on,
)


def _make_env(
    observer=None,
    trader=None,
    supplementary_observers=None,
    reward_function=None,
    feature_preprocessing_fn=None,
    **config_overrides,
) -> PolyTimeBarEnv:
    """Build an env with sensible defaults and the wait-call disabled."""
    config = PolyTimeBarEnvConfig(
        yes_token_id=config_overrides.pop("yes_token_id", "tok_yes_1"),
        action_levels=config_overrides.pop("action_levels", [-1, 0, 1]),
        **config_overrides,
    )
    env = PolyTimeBarEnv(
        config=config,
        observer=observer or MockPolymarketObserver(),
        trader=trader or MockPolymarketTrader(),
        supplementary_observers=supplementary_observers,
        reward_function=reward_function,
        feature_preprocessing_fn=feature_preprocessing_fn,
    )
    env._wait_for_next_timestamp = lambda: None
    return env


@pytest.fixture
def env():
    return _make_env(
        observer=MockPolymarketObserver(yes_price=0.72),
        trader=MockPolymarketTrader(initial_balance=10_000.0, yes_price=0.72),
        execute_on="1Hour",
    )


class TestPolyTimeBarEnv:
    """Tests for PolyTimeBarEnv."""

    def test_action_spec(self, env):
        """Action spec matches len(action_levels)."""
        assert env.action_spec.n == 3

    def test_reset_returns_required_keys_and_shapes(self, env):
        """reset() yields market_state (5,) and account_state (6,)."""
        td = env.reset()
        assert "market_state" in td.keys()
        assert "account_state" in td.keys()
        assert td["market_state"].shape == (5,)
        assert td["account_state"].shape == (6,)

    def test_account_state_initial_values(self, env):
        """After reset the agent is flat with leverage=1 and full liquidation distance."""
        td = env.reset()
        acct = td["account_state"]
        assert acct[0].item() == pytest.approx(0.0)  # exposure_pct
        assert acct[1].item() == pytest.approx(0.0)  # position_direction
        assert acct[4].item() == pytest.approx(1.0)  # leverage
        assert acct[5].item() == pytest.approx(1.0)  # distance_to_liquidation

    @pytest.mark.parametrize(
        "action_idx,expected_direction",
        [
            (0, -1),
            (1, 0),
            (2, 1),
        ],
        ids=["buy-no", "flat", "buy-yes"],
    )
    def test_step_action_direction(self, env, action_idx, expected_direction):
        """Each action_level produces the matching position_direction in account_state."""
        env.reset()
        td_out = env._step(TensorDict({"action": torch.tensor(action_idx)}, batch_size=()))
        for k in ("reward", "done", "terminated", "truncated"):
            assert k in td_out.keys()
        direction = td_out["account_state"][1].item()
        assert direction == pytest.approx(float(expected_direction), abs=0.1)

    def test_step_flat_when_already_flat_does_not_trade(self, env):
        """A flat action while flat must not change wallet balance."""
        env.reset()
        before = env.trader.get_balance()
        env._step(TensorDict({"action": torch.tensor(1)}, batch_size=()))  # action_levels[1] = 0.0
        assert env.trader.get_balance() == pytest.approx(before)

    def test_termination_on_market_close(self, env):
        """Episode terminates when the underlying market resolves."""
        env.reset()
        env.observer.market_closed = True
        td_out = env._step(TensorDict({"action": torch.tensor(1)}, batch_size=()))
        assert td_out["terminated"].item() is True

    def test_max_steps_truncation(self):
        """Episode truncates (not terminates) after max_steps; done==True."""
        env = _make_env(max_steps=2)
        env.reset()
        td1 = env._step(TensorDict({"action": torch.tensor(1)}, batch_size=()))
        assert td1["truncated"].item() is False
        td2 = env._step(TensorDict({"action": torch.tensor(1)}, batch_size=()))
        assert td2["truncated"].item() is True
        assert td2["terminated"].item() is False
        assert td2["done"].item() is True

    def test_bankruptcy_termination(self):
        """Falling below bankrupt_threshold * initial_balance terminates the episode."""
        env = _make_env(
            trader=MockPolymarketTrader(initial_balance=1.0, yes_price=0.72),
            done_on_bankruptcy=True,
            bankrupt_threshold=0.5,
        )
        env.reset()
        env._initial_balance = 10_000.0  # simulate having started rich
        td = env._step(TensorDict({"action": torch.tensor(1)}, batch_size=()))
        assert td["terminated"].item() is True

    def test_missing_market_id_raises(self):
        """ValueError when no slug / condition_id / yes_token_id is provided."""
        with pytest.raises(ValueError, match="market identifier"):
            PolyTimeBarEnv(
                config=PolyTimeBarEnvConfig(),
                observer=MockPolymarketObserver(),
                trader=MockPolymarketTrader(),
            )

    def test_close_position_on_reset(self):
        """close_position_on_reset=True closes the open YES leg before next obs."""
        env = _make_env(close_position_on_reset=True, close_position_on_init=False)
        env.reset()
        env._step(TensorDict({"action": torch.tensor(2)}, batch_size=()))  # buy YES
        env.reset()
        assert env.position.current_position == 0.0


class TestSupplementaryObservers:
    def test_specs_and_observations_merge(self):
        class Supp:
            def get_observation_spec(self):
                return {
                    "extra_data": Bounded(
                        low=-torch.inf, high=torch.inf, shape=(3,), dtype=torch.float32
                    )
                }

            def get_observations(self):
                return {"extra_data": np.array([1.0, 2.0, 3.0], dtype=np.float32)}

        env = _make_env(supplementary_observers=[Supp()])
        assert "extra_data" in env.observation_spec.keys()
        td = env.reset()
        assert td["extra_data"].shape == (3,)

    def test_key_collision_raises(self):
        class Colliding:
            def get_observation_spec(self):
                return {
                    "market_state": Bounded(
                        low=0, high=1, shape=(5,), dtype=torch.float32
                    )
                }

            def get_observations(self):
                return {"market_state": np.zeros(5, dtype=np.float32)}

        with pytest.raises(ValueError, match="collision"):
            _make_env(supplementary_observers=[Colliding()])


class TestExecuteTradeIfNeeded:
    """Coverage for the position-management state machine."""

    def test_direction_flip_closes_yes_then_opens_no(self, env):
        """Flipping from YES to NO closes the YES leg first, then opens NO."""
        env.reset()
        env._step(TensorDict({"action": torch.tensor(2)}, batch_size=()))  # buy YES
        assert env.position.current_position == 1.0
        td = env._step(TensorDict({"action": torch.tensor(0)}, batch_size=()))  # flip to NO
        assert env.position.current_position == -1.0
        assert td["account_state"][1].item() == pytest.approx(-1.0)

    def test_going_flat_from_position_calls_close(self, env):
        """Action 0.0 while holding a position closes that position."""
        env.reset()
        env._step(TensorDict({"action": torch.tensor(2)}, batch_size=()))  # buy YES
        assert env.position.current_position == 1.0
        env._step(TensorDict({"action": torch.tensor(1)}, batch_size=()))  # flat
        assert env.position.current_position == 0.0

    def test_repeated_full_action_does_not_double_buy(self):
        """Submitting the same fraction twice should produce no new trade."""
        env = _make_env(action_levels=[0.0, 1.0])  # flat, full long
        env.reset()
        env._step(TensorDict({"action": torch.tensor(1)}, batch_size=()))
        size_after_first = env.position.position_size
        env._step(TensorDict({"action": torch.tensor(1)}, batch_size=()))
        # delta_value drops below the $1 floor → no extra shares
        assert env.position.position_size == pytest.approx(size_after_first)

    def test_scaling_in_preserves_entry_price(self):
        """Going from 0.5 → 1.0 of portfolio adds shares but keeps the original entry_price."""
        env = _make_env(action_levels=[0.0, 0.5, 1.0])
        env.reset()
        env._step(TensorDict({"action": torch.tensor(1)}, batch_size=()))  # 0.5
        size_after_half = env.position.position_size
        entry_after_half = env.position.entry_price
        env._step(TensorDict({"action": torch.tensor(2)}, batch_size=()))  # 1.0
        assert env.position.position_size > size_after_half
        assert env.position.entry_price == pytest.approx(entry_after_half)


class TestAccountState:
    def test_unrealized_pnl_after_price_move(self, env):
        """account_state[2] reflects (current - entry) / entry * direction."""
        env.reset()
        env._step(TensorDict({"action": torch.tensor(2)}, batch_size=()))  # buy YES @ 0.72
        env.observer.yes_price = 0.80
        td = env._step(TensorDict({"action": torch.tensor(2)}, batch_size=()))
        expected = (0.80 - 0.72) / 0.72
        assert td["account_state"][2].item() == pytest.approx(expected, abs=1e-3)

    def test_exposure_pct_uses_current_price_not_entry_price(self, env):
        """exposure_pct must mark-to-market: numerator and denominator share the same price."""
        env.reset()
        env._step(TensorDict({"action": torch.tensor(2)}, batch_size=()))  # buy YES @ 0.72
        env.observer.yes_price = 0.50  # price drops
        td = env._step(TensorDict({"action": torch.tensor(2)}, batch_size=()))  # hold
        size = env.position.position_size
        portfolio = env._get_portfolio_value()
        expected = (size * 0.50) / portfolio
        assert td["account_state"][0].item() == pytest.approx(expected, rel=1e-3)


class TestActionResolution:
    """Defensive handling of malformed actions, mirroring OKX env."""

    def test_out_of_range_action_clamps(self, env, caplog):
        env.reset()
        with caplog.at_level(logging.WARNING):
            td = env._step(TensorDict({"action": torch.tensor(99)}, batch_size=()))
        assert td["account_state"].shape == (6,)
        assert any("out of range" in rec.message for rec in caplog.records)

    def test_missing_action_defaults_to_flat(self, env):
        """No 'action' key → default to flat index, no trade."""
        env.reset()
        before = env.trader.get_balance()
        td = env._step(TensorDict({}, batch_size=()))
        assert env.trader.get_balance() == pytest.approx(before)
        assert td["account_state"][1].item() == pytest.approx(0.0)


class TestCloseLifecycle:
    def test_close_logs_warning_when_position_open(self, env, caplog):
        env.reset()
        env._step(TensorDict({"action": torch.tensor(2)}, batch_size=()))  # buy YES
        with caplog.at_level(logging.WARNING):
            env.close()
        assert any("open position" in rec.message for rec in caplog.records)

    def test_close_silent_when_flat(self, env, caplog):
        env.reset()
        with caplog.at_level(logging.WARNING):
            env.close()
        assert not any("open position" in rec.message for rec in caplog.records)


@pytest.mark.parametrize(
    "value,expected",
    [
        ("1Hour", (1, "Hour")),
        ("5Minute", (5, "Minute")),
        ("1Day", (1, "Day")),
        ("invalid", (1, "Hour")),  # silent fallback
        ("Hour", (1, "Hour")),     # missing count → fallback
    ],
    ids=["hour", "minute", "day", "garbage", "no-count"],
)
def test_parse_execute_on(value, expected):
    """Directly exercises the parser without env construction overhead."""
    assert _parse_execute_on(value) == expected


class TestRealConstruction:
    """Smoke tests for the lazy observer/trader construction path."""

    def test_constructs_with_default_observer_and_trader(self, monkeypatch):
        fake_observer_cls = MagicMock(return_value=MockPolymarketObserver())
        fake_trader_cls = MagicMock(return_value=MockPolymarketTrader())
        monkeypatch.setattr(
            "torchtrade.envs.live.polymarket.observation.PolymarketObservationClass",
            fake_observer_cls,
        )
        monkeypatch.setattr(
            "torchtrade.envs.live.polymarket.order_executor.PolymarketOrderExecutor",
            fake_trader_cls,
        )

        config = PolyTimeBarEnvConfig(
            yes_token_id="tok_yes_1",
            close_position_on_init=False,
            dry_run=True,
        )
        env = PolyTimeBarEnv(config=config, private_key="0xkey")
        env._wait_for_next_timestamp = lambda: None

        fake_observer_cls.assert_called_once()
        fake_trader_cls.assert_called_once()
        assert fake_trader_cls.call_args.kwargs["private_key"] == "0xkey"
        assert fake_trader_cls.call_args.kwargs["dry_run"] is True

        td = env.reset()
        assert td["market_state"].shape == (5,)
        assert td["account_state"].shape == (6,)


class TestCloseOnInit:
    def test_close_position_on_init_true_invokes_trader(self):
        """The default config closes existing positions during construction."""
        trader = MockPolymarketTrader()
        trader.cancel_all = MagicMock(wraps=trader.cancel_all)
        trader.close_position = MagicMock(wraps=trader.close_position)

        # Default config has close_position_on_init=True.
        env = _make_env(trader=trader, close_position_on_init=True)
        assert trader.cancel_all.called
        # close_position called once for YES and once for NO.
        called_tokens = {c.args[0] for c in trader.close_position.call_args_list}
        assert "tok_yes_1" in called_tokens
        assert "tok_no_1" in called_tokens

    def test_close_position_on_init_false_does_not_call_trader(self):
        trader = MockPolymarketTrader()
        trader.cancel_all = MagicMock(wraps=trader.cancel_all)
        trader.close_position = MagicMock(wraps=trader.close_position)

        _make_env(trader=trader, close_position_on_init=False)
        assert not trader.cancel_all.called
        assert not trader.close_position.called


class TestPluggableHooks:
    def test_feature_preprocessing_fn_forwarded_to_default_observer(self, monkeypatch):
        """env wires feature_preprocessing_fn into the lazily-built default observer."""
        seen = {}

        def capture(yes_token_id, market_slug, condition_id, feature_preprocessing_fn):
            seen["fn"] = feature_preprocessing_fn
            return MockPolymarketObserver()

        monkeypatch.setattr(
            "torchtrade.envs.live.polymarket.observation.PolymarketObservationClass",
            capture,
        )

        sentinel = lambda x: x  # noqa: E731
        config = PolyTimeBarEnvConfig(
            yes_token_id="tok_yes_1", close_position_on_init=False, dry_run=True
        )
        env = PolyTimeBarEnv(
            config=config,
            trader=MockPolymarketTrader(),
            feature_preprocessing_fn=sentinel,
        )
        env._wait_for_next_timestamp = lambda: None
        assert seen["fn"] is sentinel

    def test_custom_reward_function_is_called_with_history(self):
        """env._step passes its HistoryTracker to a user-supplied reward function."""
        sentinel = MagicMock(return_value=0.42)
        env = _make_env(reward_function=sentinel)
        env.reset()
        td = env._step(TensorDict({"action": torch.tensor(1)}, batch_size=()))
        sentinel.assert_called_once_with(env.history)
        assert td["reward"].item() == pytest.approx(0.42)
