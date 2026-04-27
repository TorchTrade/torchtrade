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
        # Stub the polling loop directly, bypassing _fetch_resolved_outcome's
        # network call AND the time.sleep inside _poll_for_resolution.
        env._poll_for_resolution = MagicMock(
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
        "action,outcome,fill_price,expected_payoff",
        [
            (1, 1, 0.4, 100.0 * 0.6 / 0.4),    # bet Up @ 0.4, Up wins  → +150
            (0, 0, 0.6, 100.0 * 0.4 / 0.6),    # bet Down @ 0.6, Down wins → +66.67
            (1, 0, None, -100.0),               # bet Up, Down wins → -stake
            (0, 1, None, -100.0),               # bet Down, Up wins → -stake
        ],
        ids=["up-wins", "down-wins", "up-loses", "down-loses"],
    )
    def test_payoff_path(self, action, outcome, fill_price, expected_payoff):
        """Pin the exact payoff (not just sign) so a fill-price swap on action=0
        — e.g. picking ``yes_price`` when the agent bet Down — would still pass
        a sign-only check but produce the wrong magnitude. ``stake = 1000 * 0.1
        = 100`` per ``_make_env`` defaults.
        """
        market = _make_market(yes_price=0.4, no_price=0.6)
        env, _, _ = _make_env(outcomes=[outcome], markets=[market, _make_market()])
        env.reset()
        td = env._step(TensorDict({"action": torch.tensor(action)}, batch_size=()))
        assert td["reward"].item() == pytest.approx(expected_payoff)

    def test_dry_run_skips_trader_buy(self):
        env, _, trader = _make_env(config_overrides={"dry_run": True})
        env.reset()
        env._step(TensorDict({"action": torch.tensor(1)}, batch_size=()))
        trader.buy.assert_not_called()

    @pytest.mark.parametrize(
        "action,expected_token_id",
        [(1, "tok_yes"), (0, "tok_no")],
        ids=["bet-up-targets-yes-token", "bet-down-targets-no-token"],
    )
    def test_live_mode_calls_trader_buy_with_correct_token(self, action, expected_token_id):
        """A YES↔NO token swap on either action would silently pass without
        parametrizing both directions — so this test exercises both."""
        env, _, trader = _make_env(config_overrides={"dry_run": False})
        env.reset()
        env._step(TensorDict({"action": torch.tensor(action)}, batch_size=()))
        trader.buy.assert_called_once()
        kwargs = trader.buy.call_args.kwargs
        assert kwargs["token_id"] == expected_token_id
        assert kwargs["amount_usdc"] > 0

    def test_zero_stake_in_live_mode_does_not_call_trader(self):
        """bet_fraction=0 → stake=0 → no order even in live mode."""
        env, _, trader = _make_env(
            config_overrides={"dry_run": False, "bet_fraction": 0.0}
        )
        env.reset()
        env._step(TensorDict({"action": torch.tensor(1)}, batch_size=()))
        trader.buy.assert_not_called()

    @pytest.mark.parametrize(
        "underlying_outcome",
        [1, 0],
        ids=["would-have-won", "would-have-lost"],
    )
    def test_failed_order_in_live_mode_books_zero_payoff(self, underlying_outcome):
        """Critical safety: a rejected/failed order must NOT produce phantom P&L.

        Pins the FULL post-failure contract, not just reward, so a future
        refactor that ``return``-s early on failure (skipping the next-market
        fetch, step counter, or done flags) breaks the test instead of silently
        breaking episode progression. Parametrized over both possible underlying
        outcomes, a regression that booked ``-stake`` on failure (instead of 0)
        would only show up in the would-have-lost case otherwise.
        """
        market = _make_market(yes_price=0.4, no_price=0.6)
        env, scanner, trader = _make_env(
            outcomes=[underlying_outcome],
            markets=[market, _make_market(slug="btc-updown-5m-next")],
            config_overrides={"dry_run": False},
        )
        trader.buy.return_value = {"success": False, "error": "insufficient USDC"}
        env.reset()
        cash_before = env.cash
        scanner_calls_before = scanner.next_active_market.call_count
        td = env._step(TensorDict({"action": torch.tensor(1)}, batch_size=()))

        # We tried to place the order
        trader.buy.assert_called_once()

        # No phantom P&L
        assert td["reward"].item() == 0.0
        assert env.cash == pytest.approx(cash_before)

        # Episode progression continues normally, these assertions catch a
        # naive `if failure: return early_td` shortcut.
        assert env._step_count == 1
        assert not td["terminated"].item()
        assert not td["truncated"].item()
        assert scanner.next_active_market.call_count > scanner_calls_before
        # Next market's state is non-zero (real market_state, not the terminal
        # zero-tensor used for "no next market" cases).
        assert not torch.equal(td["market_state"], torch.zeros(4))

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
    """Reads the CLOB midpoint per outcome token, not Gamma."""

    @pytest.mark.parametrize(
        "yes_mid,no_mid,expected",
        [
            ("1.0", "0.0", 1),       # Up snapped fully
            ("0.0", "1.0", 0),       # Down snapped fully
            ("0.995", "0.005", 1),   # boundary: just at threshold (Up)
            ("0.005", "0.995", 0),   # boundary: just at threshold (Down)
            ("0.989", "0.011", None),  # below the 0.99 threshold
            ("0.99", "0.05", None),    # Up at threshold but Down still too high
            ("0.5", "0.5", None),      # mid-market
        ],
        ids=["up", "down", "tight-up", "tight-down", "below-up",
             "asymmetric-pending", "midmarket"],
    )
    def test_outcome_parsing(self, yes_mid, no_mid, expected):
        env, _, _ = _make_env(mock_fetch=False)
        m = _make_market()
        with patch("torchtrade.envs.live.polymarket.env.requests.get") as mock_get:
            def respond(url, params=None, **_):
                resp = MagicMock()
                if params["token_id"] == m.yes_token_id:
                    resp.json.return_value = {"mid": yes_mid}
                else:
                    resp.json.return_value = {"mid": no_mid}
                resp.raise_for_status = MagicMock()
                return resp
            mock_get.side_effect = respond
            assert env._fetch_resolved_outcome(m) == expected

    def test_outgoing_request_pins_clob_endpoint_and_token_id(self):
        """Pin the CLOB API contract, endpoint URL and token_id query param.
        A regression that flipped to Gamma's outcomePrices would fail this."""
        env, _, _ = _make_env(mock_fetch=False)
        m = _make_market()
        with patch("torchtrade.envs.live.polymarket.env.requests.get") as mock_get:
            mock_resp = MagicMock()
            mock_resp.json.return_value = {"mid": "0.5"}
            mock_resp.raise_for_status = MagicMock()
            mock_get.return_value = mock_resp
            env._fetch_resolved_outcome(m)
        # Two calls, one per outcome token
        assert mock_get.call_count == 2
        urls = [c.args[0] for c in mock_get.call_args_list]
        assert all("clob.polymarket.com" in u for u in urls)
        assert all("/midpoint" in u for u in urls)
        token_ids = {c.kwargs["params"]["token_id"] for c in mock_get.call_args_list}
        assert token_ids == {m.yes_token_id, m.no_token_id}

    def test_http_failure_returns_none(self):
        import requests
        env, _, _ = _make_env(mock_fetch=False)
        with patch(
            "torchtrade.envs.live.polymarket.env.requests.get",
            side_effect=requests.ConnectionError("503"),
        ):
            assert env._fetch_resolved_outcome(_make_market()) is None

    def test_missing_mid_field_returns_none(self):
        """If the CLOB response shape changes (no 'mid' key), don't crash."""
        env, _, _ = _make_env(mock_fetch=False)
        with patch("torchtrade.envs.live.polymarket.env.requests.get") as mock_get:
            mock_resp = MagicMock()
            mock_resp.json.return_value = {"unexpected": "shape"}
            mock_resp.raise_for_status = MagicMock()
            mock_get.return_value = mock_resp
            assert env._fetch_resolved_outcome(_make_market()) is None


class TestPollForResolution:
    """Polling loop must retry until Gamma reports a resolved outcome."""

    def _real_poll_env(self, **config_overrides):
        env, _, _ = _make_env(config_overrides=config_overrides or None)
        # Restore the real polling loop (the fixture stubs it for other tests).
        env._poll_for_resolution = (
            PolymarketBetEnv._poll_for_resolution.__get__(env, PolymarketBetEnv)
        )
        return env

    def test_returns_first_non_none_outcome_immediately(self):
        env = self._real_poll_env()
        env._fetch_resolved_outcome = MagicMock(return_value=1)
        with patch("torchtrade.envs.live.polymarket.env.time.sleep") as mock_sleep:
            assert env._poll_for_resolution("0xcond") == 1
        mock_sleep.assert_not_called()
        assert env._fetch_resolved_outcome.call_count == 1

    def test_retries_until_resolved(self):
        env = self._real_poll_env()
        # First two calls: still mid-market. Third: Up wins.
        env._fetch_resolved_outcome = MagicMock(side_effect=[None, None, 1])
        with patch("torchtrade.envs.live.polymarket.env.time.sleep"):
            assert env._poll_for_resolution("0xcond") == 1
        assert env._fetch_resolved_outcome.call_count == 3

    def test_returns_none_when_max_wait_exhausted(self):
        env = self._real_poll_env(
            resolution_max_wait_seconds=0.1,
            resolution_poll_interval_seconds=0.01,
        )
        env._fetch_resolved_outcome = MagicMock(return_value=None)
        with patch("torchtrade.envs.live.polymarket.env.time.sleep"):
            assert env._poll_for_resolution("0xcond") is None
        assert env._fetch_resolved_outcome.call_count >= 1


class TestWaitForResolution:
    """The real method is mocked out in _make_env, verify its branches separately."""

    @pytest.mark.parametrize(
        "end_date_value,expect_sleep",
        [
            ("", False),
            ("garbage", False),
            ("2000-01-01T00:00:00Z", False),  # past, sleep_seconds <= 0, no sleep
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
