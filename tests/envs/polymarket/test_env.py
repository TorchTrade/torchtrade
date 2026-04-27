"""Tests for PolymarketBetEnv."""

import itertools
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest
import torch
from tensordict import TensorDict
from torchrl.envs.utils import check_env_specs

from torchtrade.envs.live.polymarket.env import (
    PolymarketBetEnv,
    PolymarketBetEnvConfig,
)
from torchtrade.envs.live.polymarket.market_scanner import PolymarketMarket


def _make_market(
    yes_price: float = 0.5,
    no_price: float = 0.5,
    end_in_seconds: float = 300.0,
    slug: str = "btc-updown-5m-1234",
    condition_id: str = "0xcond1",
    spread: float = 0.02,
    volume_24h: float = 1500.0,
    liquidity: float = 12000.0,
) -> PolymarketMarket:
    end = datetime.now(timezone.utc) + timedelta(seconds=end_in_seconds)
    return PolymarketMarket(
        market_id="m1",
        condition_id=condition_id,
        question="Bitcoin Up or Down",
        description="",
        slug=slug,
        yes_token_id="tok_yes",
        no_token_id="tok_no",
        yes_price=yes_price,
        no_price=no_price,
        volume_24h=volume_24h,
        total_volume=volume_24h * 10,
        liquidity=liquidity,
        spread=spread,
        end_date=end.isoformat().replace("+00:00", "Z"),
        tags=[],
        neg_risk=False,
    )


def _make_env(
    *,
    outcomes=None,                      # iterable of resolved outcomes per step (default: forever Up)
    markets=None,                       # iterable of next markets (default: forever fresh)
    mock_fetch=True,                    # if False, leave _fetch_resolved_outcome real
    config_overrides=None,
):
    """Build an env with stubbed scanner/trader/wait/resolve. Returns (env, scanner, trader)."""
    cfg_kwargs = {
        "market_slug_prefix": "btc-updown-5m-",
        "max_steps": 5,
        "initial_cash": 1_000.0,
        "bet_fraction": 0.1,
        "dry_run": True,
    }
    cfg_kwargs.update(config_overrides or {})
    config = PolymarketBetEnvConfig(**cfg_kwargs)

    scanner = MagicMock()
    if markets is None:
        scanner.next_active_market.side_effect = lambda *_a, **_k: _make_market()
    else:
        scanner.next_active_market.side_effect = list(markets)

    trader = MagicMock()
    trader.buy.return_value = {"success": True}

    env = PolymarketBetEnv(config, scanner=scanner, trader=trader)
    env._wait_for_resolution = lambda *a, **k: None
    if mock_fetch:
        outcome_iter = (
            itertools.cycle([1]) if outcomes is None else iter(outcomes)
        )
        env._fetch_resolved_outcome = MagicMock(
            side_effect=lambda *_a, **_k: next(outcome_iter)
        )
    return env, scanner, trader


# --- Spec / construction ---------------------------------------------------- #

class TestSpecs:
    def test_check_env_specs_passes(self):
        env, _, _ = _make_env()
        check_env_specs(env)

    def test_observation_has_only_market_state(self):
        env, _, _ = _make_env()
        assert list(env.observation_spec.keys()) == ["market_state"]
        assert env.observation_spec["market_state"].shape == (4,)

    def test_action_is_binary(self):
        env, _, _ = _make_env()
        assert env.action_spec.n == 2

    def test_missing_slug_prefix_raises(self):
        with pytest.raises(ValueError, match="market_slug_prefix"):
            PolymarketBetEnv(PolymarketBetEnvConfig())

    def test_default_trader_constructed_when_not_injected(self):
        """Lazy import path: ``PolymarketOrderExecutor`` is built when no trader is passed."""
        config = PolymarketBetEnvConfig(
            market_slug_prefix="btc-updown-5m-", dry_run=True
        )
        scanner = MagicMock()
        scanner.next_active_market.return_value = _make_market()
        env = PolymarketBetEnv(config, scanner=scanner)
        from torchtrade.envs.live.polymarket.order_executor import (
            PolymarketOrderExecutor,
        )
        assert isinstance(env.trader, PolymarketOrderExecutor)


# --- Pure helpers ----------------------------------------------------------- #

class TestComputePayoff:
    @pytest.mark.parametrize(
        "action,fill,outcome,expected",
        [
            (1, 0.4, 1, 100.0 * 0.6 / 0.4),
            (0, 0.4, 0, 100.0 * 0.6 / 0.4),
            (1, 0.4, 0, -100.0),
            (0, 0.4, 1, -100.0),
            (1, 0.5, 1, 100.0),
        ],
        ids=["up-wins", "down-wins", "up-loses", "down-loses", "even-money"],
    )
    def test_payoff(self, action, fill, outcome, expected):
        assert PolymarketBetEnv._compute_payoff(action, fill, outcome, 100.0) == pytest.approx(expected)


# --- Reset ------------------------------------------------------------------ #

class TestReset:
    def test_returns_market_state_and_done_flags(self):
        env, _, _ = _make_env()
        td = env.reset()
        assert td["market_state"].shape == (4,)
        assert not td["terminated"].item()
        assert not td["truncated"].item()

    def test_resets_cash_and_step_counter(self):
        env, _, _ = _make_env()
        env.cash = 0.42
        env._step_count = 99
        env.reset()
        assert env.cash == env.config.initial_cash
        assert env._step_count == 0

    def test_raises_when_no_markets(self):
        env, scanner, _ = _make_env()
        scanner.next_active_market.side_effect = [None]
        with pytest.raises(RuntimeError, match="No active markets"):
            env.reset()


# --- Step ------------------------------------------------------------------- #

class TestStep:
    @pytest.mark.parametrize(
        "action,outcome,stake_won",
        [
            (1, 1, True),   # bet Up, Up wins
            (0, 0, True),   # bet Down, Down wins
            (1, 0, False),  # bet Up, Down wins
            (0, 1, False),  # bet Down, Up wins
        ],
        ids=["up-wins", "down-wins", "up-loses", "down-loses"],
    )
    def test_payoff_path(self, action, outcome, stake_won):
        market = _make_market(yes_price=0.4, no_price=0.6)
        env, _, _ = _make_env(outcomes=[outcome], markets=[market, _make_market()])
        env.reset()
        td = env._step(TensorDict({"action": torch.tensor(action)}, batch_size=()))
        if stake_won:
            assert td["reward"].item() > 0
        else:
            assert td["reward"].item() < 0

    def test_dry_run_skips_trader_buy(self):
        env, _, trader = _make_env(config_overrides={"dry_run": True})
        env.reset()
        env._step(TensorDict({"action": torch.tensor(1)}, batch_size=()))
        trader.buy.assert_not_called()

    def test_live_mode_calls_trader_buy(self):
        env, _, trader = _make_env(config_overrides={"dry_run": False})
        env.reset()
        env._step(TensorDict({"action": torch.tensor(1)}, batch_size=()))
        trader.buy.assert_called_once()
        kwargs = trader.buy.call_args.kwargs
        assert kwargs["token_id"] == "tok_yes"
        assert kwargs["amount_usdc"] > 0

    def test_zero_stake_in_live_mode_does_not_call_trader(self):
        """bet_fraction=0 → stake=0 → no order even in live mode."""
        env, _, trader = _make_env(
            config_overrides={"dry_run": False, "bet_fraction": 0.0}
        )
        env.reset()
        env._step(TensorDict({"action": torch.tensor(1)}, batch_size=()))
        trader.buy.assert_not_called()

    def test_failed_order_in_live_mode_books_zero_payoff(self):
        """Critical safety: a rejected/failed order must NOT produce phantom P&L.

        Previously the env discarded ``trader.buy()``'s return value and computed
        payoff against the assumed fill — so a FOK rejection or insufficient-USDC
        error would still be "won" or "lost" against a position that never existed.
        """
        market = _make_market(yes_price=0.4, no_price=0.6)
        env, _, trader = _make_env(
            outcomes=[1],                       # market resolves Up
            markets=[market, _make_market()],
            config_overrides={"dry_run": False},
        )
        # Force the trader to report failure
        trader.buy.return_value = {"success": False, "error": "insufficient USDC"}
        env.reset()
        cash_before = env.cash
        td = env._step(TensorDict({"action": torch.tensor(1)}, batch_size=()))

        trader.buy.assert_called_once()         # we did try to place
        assert td["reward"].item() == 0.0       # but no phantom payoff
        assert env.cash == pytest.approx(cash_before)

    def test_cash_updates_after_win_and_loss(self):
        env, _, _ = _make_env(
            outcomes=[1, 0],
            markets=[
                _make_market(yes_price=0.4, no_price=0.6),
                _make_market(yes_price=0.4, no_price=0.6),
                _make_market(),
            ],
        )
        env.reset()
        before = env.cash
        env._step(TensorDict({"action": torch.tensor(1)}, batch_size=()))  # Up wins
        after_win = env.cash
        env._step(TensorDict({"action": torch.tensor(1)}, batch_size=()))  # Up loses
        assert after_win > before
        assert env.cash < after_win

    def test_max_steps_truncation(self):
        env, _, _ = _make_env(
            outcomes=[1, 1],
            config_overrides={"max_steps": 2, "done_on_bankruptcy": False},
        )
        env.reset()
        td1 = env._step(TensorDict({"action": torch.tensor(1)}, batch_size=()))
        assert not td1["truncated"].item()
        td2 = env._step(TensorDict({"action": torch.tensor(1)}, batch_size=()))
        assert td2["truncated"].item()
        assert not td2["terminated"].item()

    def test_bankruptcy_termination(self):
        market = _make_market(yes_price=0.5, no_price=0.5)
        env, _, _ = _make_env(
            outcomes=[0],
            markets=[market, _make_market()],
            config_overrides={
                "initial_cash": 10.0,
                "bet_fraction": 1.0,
                "done_on_bankruptcy": True,
                "bankrupt_threshold": 0.5,
            },
        )
        env.reset()
        td = env._step(TensorDict({"action": torch.tensor(1)}, batch_size=()))
        assert td["terminated"].item()

    def test_terminated_wins_when_bankruptcy_and_max_steps_coincide(self):
        """If both fire on the same step, terminated suppresses truncated."""
        market = _make_market(yes_price=0.5, no_price=0.5)
        env, _, _ = _make_env(
            outcomes=[0],
            markets=[market, _make_market()],
            config_overrides={
                "initial_cash": 10.0,
                "bet_fraction": 1.0,
                "done_on_bankruptcy": True,
                "bankrupt_threshold": 0.5,
                "max_steps": 1,
            },
        )
        env.reset()
        td = env._step(TensorDict({"action": torch.tensor(1)}, batch_size=()))
        assert td["terminated"].item()
        assert not td["truncated"].item()

    def test_no_next_market_terminates_with_zero_obs(self):
        """When the scanner finds no follow-up market, terminate and emit zeros."""
        env, _, _ = _make_env(
            outcomes=[1],
            markets=[_make_market(), None],
        )
        env.reset()
        td = env._step(TensorDict({"action": torch.tensor(1)}, batch_size=()))
        assert td["terminated"].item()
        assert torch.equal(td["market_state"], torch.zeros(4))

    def test_unresolved_outcome_yields_zero_reward(self):
        env, _, _ = _make_env(outcomes=[None])
        env.reset()
        td = env._step(TensorDict({"action": torch.tensor(1)}, batch_size=()))
        assert td["reward"].item() == 0.0

    def test_step_before_reset_raises(self):
        env, _, _ = _make_env()
        with pytest.raises(RuntimeError, match="reset"):
            env._step(TensorDict({"action": torch.tensor(1)}, batch_size=()))

    def test_full_step_via_torchrl_wraps_in_next(self):
        """The standard ``env.step(td)`` API nests results under ``next``."""
        env, _, _ = _make_env()
        td = env.reset()
        out = env.step(td.set("action", torch.tensor(1)))
        assert "next" in out.keys()
        assert "reward" in out["next"].keys()


# --- Outcome parsing -------------------------------------------------------- #

class TestFetchResolvedOutcome:
    @pytest.mark.parametrize(
        "response_factory,expected",
        [
            (lambda: [{"outcomePrices": '["1.0", "0.0"]'}], 1),
            (lambda: [{"outcomePrices": '["0.0", "1.0"]'}], 0),
            (lambda: [{"outcomePrices": '["0.995", "0.005"]'}], 1),    # boundary: just at threshold
            (lambda: [{"outcomePrices": '["0.989", "0.011"]'}], None), # boundary: just below
            (lambda: [{"outcomePrices": '["0.99", "0.05"]'}], None),   # up at threshold but down too high
            (lambda: [{"outcomePrices": '["0.5", "0.5"]'}], None),
            (lambda: [{"outcomePrices": "[]"}], None),
            (lambda: [{"outcomePrices": "not-json"}], None),
            (lambda: [], None),
            (lambda: {"outcomePrices": '["1.0", "0.0"]'}, 1),  # dict response (some Gamma versions)
        ],
        ids=["up", "down", "tight-up", "tight-up-below", "tight-up-down-too-high",
             "midmarket", "empty", "malformed", "no-results", "dict-response"],
    )
    def test_outcome_parsing(self, response_factory, expected):
        env, _, _ = _make_env(mock_fetch=False)
        with patch("torchtrade.envs.live.polymarket.env.requests.get") as mock_get:
            mock_resp = MagicMock()
            mock_resp.json.return_value = response_factory()
            mock_resp.raise_for_status = MagicMock()
            mock_get.return_value = mock_resp
            assert env._fetch_resolved_outcome("0xcond") == expected

    def test_outgoing_request_pins_endpoint_and_params(self):
        """Pin the Gamma API contract — endpoint URL and condition_id query param."""
        env, _, _ = _make_env(mock_fetch=False)
        with patch("torchtrade.envs.live.polymarket.env.requests.get") as mock_get:
            mock_resp = MagicMock()
            mock_resp.json.return_value = [{"outcomePrices": '["1.0", "0.0"]'}]
            mock_resp.raise_for_status = MagicMock()
            mock_get.return_value = mock_resp
            env._fetch_resolved_outcome("0xcond_abc")
        assert "gamma-api.polymarket.com" in mock_get.call_args.args[0]
        assert "/markets" in mock_get.call_args.args[0]
        assert mock_get.call_args.kwargs["params"]["condition_id"] == "0xcond_abc"

    def test_http_failure_returns_none(self):
        import requests
        env, _, _ = _make_env(mock_fetch=False)
        with patch(
            "torchtrade.envs.live.polymarket.env.requests.get",
            side_effect=requests.ConnectionError("503"),
        ):
            assert env._fetch_resolved_outcome("0xcond") is None


class TestWaitForResolution:
    """The real method is mocked out in _make_env — verify its branches separately."""

    @pytest.mark.parametrize(
        "end_date_value,expect_sleep",
        [
            ("", False),
            ("garbage", False),
            ("2000-01-01T00:00:00Z", False),  # past — sleep_seconds <= 0, no sleep
        ],
        ids=["empty", "malformed", "past"],
    )
    def test_no_sleep_when_no_future_endDate(self, end_date_value, expect_sleep):
        env, _, _ = _make_env()
        # Restore the real method
        env._wait_for_resolution = (
            PolymarketBetEnv._wait_for_resolution.__get__(env, PolymarketBetEnv)
        )
        with patch("torchtrade.envs.live.polymarket.env.time.sleep") as mock_sleep:
            env._wait_for_resolution(end_date_value)
            assert mock_sleep.called == expect_sleep

    def test_sleeps_when_future_endDate(self):
        env, _, _ = _make_env()
        env._wait_for_resolution = (
            PolymarketBetEnv._wait_for_resolution.__get__(env, PolymarketBetEnv)
        )
        future = (datetime.now(timezone.utc) + timedelta(seconds=120)).isoformat().replace(
            "+00:00", "Z"
        )
        with patch("torchtrade.envs.live.polymarket.env.time.sleep") as mock_sleep:
            env._wait_for_resolution(future)
            mock_sleep.assert_called_once()
            slept_for = mock_sleep.call_args.args[0]
            assert slept_for > 100  # 120s + 30s grace ≈ 150s


class TestCloseLifecycle:
    def test_close_calls_trader_cancel_all(self):
        env, _, trader = _make_env()
        env.close()
        trader.cancel_all.assert_called_once()
