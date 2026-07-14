"""Tests for PolymarketBetEnv."""

import dataclasses
import itertools
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, call, patch

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
        # Skip the 30s real-time wait between env-level next-market retries so
        # tests that exercise the None path don't burn 60+ seconds each.
        "next_market_retry_sleep_seconds": 0,
    }
    cfg_kwargs.update(config_overrides or {})
    config = PolymarketBetEnvConfig(**cfg_kwargs)

    scanner = MagicMock()
    if markets is None:
        scanner.next_active_market.side_effect = lambda *_a, **_k: _make_market()
    else:
        scanner.next_active_market.side_effect = list(markets)

    trader = MagicMock()

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

    def test_market_state_is_exactly_yes_spread_volume_liquidity(self):
        """The policy's ENTIRE input. Pin the ORDER and the SOURCE FIELD of all four slots.

        Four mutually distinct values, so no swap can hide: spread<->liquidity,
        volume<->liquidity, and yes_price->no_price each survived the whole suite before this
        test existed. A permutation is invisible to check_env_specs -- Bounded(0, inf) accepts
        any order -- and feeding the policy `no_price` inverts the market it thinks it sees.
        (CLAUDE.md invariant #3: an observation mutation once passed 1977 tests.)
        """
        market = _make_market(yes_price=0.42, no_price=0.58, spread=0.02,
                              volume_24h=1500.0, liquidity=12000.0)
        env, _, _ = _make_env(markets=[market])
        td = env.reset()
        assert torch.equal(
            td["market_state"],
            torch.tensor([0.42, 0.02, 1500.0, 12000.0], dtype=torch.float32),
        )

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
        # Pins that the env actually wires the executor to PAPER rather than trusting the
        # executor's default. Belt-and-braces: the executor now defaults to dry_run=True too,
        # so BOTH must regress before a live client can be built -- this assertion is what
        # makes that take two mistakes instead of one.
        assert env.trader._dry_run is True


# --- Pure helpers ----------------------------------------------------------- #

class TestClose:
    def test_close_works_through_a_transform(self):
        """Regression: `def close(self)` did not accept EnvBase's keyword-only
        `raise_if_closed`, which TransformedEnv forwards -- so closing a wrapped env raised
        TypeError. RewardSum is the wrapper this env's own spec comment points users at."""
        from torchrl.envs import TransformedEnv
        from torchrl.envs.transforms import RewardSum

        env, _, trader = _make_env()
        TransformedEnv(env, RewardSum()).close()   # must not raise
        trader.cancel_all.assert_called_once()
        # NOT asserting is_closed: EnvBase.close() only sets is_closed=True, and this env's
        # is_closed is ALREADY True from construction -- so the assertion would pass whether or
        # not close() delegates upward. A dead assertion is worse than no assertion. The
        # super() call in close() is correct delegation, but it is currently unobservable.


class TestComputePayoff:
    @pytest.mark.parametrize(
        "action,fill,outcome,expected",
        [
            (1, 0.4, 1, 100.0 * 0.6 / 0.4),
            (0, 0.4, 0, 100.0 * 0.6 / 0.4),
            (1, 0.4, 0, -100.0),
            (0, 0.4, 1, -100.0),
            (1, 0.5, 1, 100.0),
            # A market quoting 0.0 must not ZeroDivisionError mid-episode. The chosen silent
            # default -- a WIN at fill 0 books 0.0, not an infinite payout -- is pinned here
            # because deleting the guard otherwise passes the whole suite.
            (1, 0.0, 1, 0.0),
        ],
        ids=["up-wins", "down-wins", "up-loses", "down-loses", "even-money", "zero-fill-price"],
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
        """Reset still raises after the env-level retry budget is exhausted."""
        env, scanner, _ = _make_env()
        scanner.next_active_market.side_effect = [None] * env.config.next_market_max_attempts
        with pytest.raises(RuntimeError, match="No active markets"):
            env.reset()
        assert scanner.next_active_market.call_count == env.config.next_market_max_attempts


# --- Step ------------------------------------------------------------------- #

class TestStep:
    def test_step_returns_the_NEXT_market_state_not_the_resolved_one(self):
        """_step's observation must come from the market the agent will BET ON next.

        The reset path was pinned; this one was not, and `_market_state(next_market)` ->
        `_market_state(market)` (the stale, already-resolved market) passed the whole suite.
        Live that is severe: the policy conditions on the OLD market's quotes while the next
        bet's fill_price is read from the NEW one -- the agent bets at prices it never saw.
        The two markets here differ in every slot so no swap or staleness can hide.
        (CLAUDE.md invariant #3, one method over from where it was just fixed.)
        """
        resolved = _make_market(yes_price=0.42, spread=0.02, volume_24h=1500.0, liquidity=12000.0)
        upcoming = _make_market(yes_price=0.77, spread=0.05, volume_24h=999.0, liquidity=4321.0)
        env, _, _ = _make_env(outcomes=[1], markets=[resolved, upcoming])
        env.reset()
        td = env._step(TensorDict({"action": torch.tensor(1)}, batch_size=()))
        assert torch.equal(
            td["market_state"],
            torch.tensor([0.77, 0.05, 999.0, 4321.0], dtype=torch.float32),
        )

    def test_step_waits_for_endDate_before_polling(self):
        """The wait must happen BEFORE the poll, and be handed the right market.

        Nothing observed this: _make_env stubs _wait_for_resolution with a bare lambda, so
        deleting the call, or polling first, both passed. Live, polling a market that has not
        ended means the midpoint never snaps to 0.99/0.01, the resolution budget burns out, and
        EVERY bet books reward=0 -- silent in tests, poison in the training data.
        Also pins that _poll_for_resolution gets the MARKET, not market.condition_id (which
        would AttributeError on the first live step).
        """
        market = _make_market()
        env, _, _ = _make_env(outcomes=[1], markets=[market, _make_market()], mock_fetch=False)
        env.reset()
        rec = MagicMock()
        env._wait_for_resolution = rec.wait
        env._poll_for_resolution = lambda *a, **k: (rec.poll(*a, **k), 1)[1]
        env._step(TensorDict({"action": torch.tensor(1)}, batch_size=()))

        assert [c[0] for c in rec.mock_calls] == ["wait", "poll"]   # order, not just presence
        rec.wait.assert_called_once_with(market.end_date)
        rec.poll.assert_called_once_with(market)

    def test_paper_env_never_touches_the_trader_during_a_step(self):
        """The paper env must not call the trader AT ALL during a step -- not just `.buy`.

        Asserting `buy.assert_not_called()` pinned only that one attribute name; a _step that
        called `trader.submit_market_order(...)` would have passed. mock_calls == [] pins the
        whole surface, which is the actual contract: nothing is submitted, ever.
        """
        env, _, trader = _make_env()
        env.reset()
        env._step(TensorDict({"action": torch.tensor(1)}, batch_size=()))
        assert trader.mock_calls == []

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

        # EXACT values, not just direction: stake COMPOUNDS off current cash, and a fixed stake
        # (initial_cash * frac) satisfies "up then down" just as well, so directional asserts
        # cannot tell the two apart. This is the rule that decides how much is at risk per bet.
        # bet_fraction=0.1, fill=0.4 -> win pays stake*(1-f)/f = 1.5x stake.
        #   stake1 = 1000 * 0.1 = 100      -> +150  -> 1150
        #   stake2 = 1150 * 0.1 = 115      -> -115  -> 1035   (a fixed stake would give 1050)
        assert before == pytest.approx(1000.0)
        assert after_win == pytest.approx(1150.0)
        assert env.cash == pytest.approx(1035.0)

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

    @pytest.mark.parametrize("cash,enabled,expected", [
        # The two rows that pin the WIRING. The boundary and the arithmetic belong to the shared
        # rule and are pinned in tests/envs/test_termination.py -- asserting them again through
        # the env is the layering this test exists to avoid.
        (199.99, True, True),   # sole killer of the bet_fraction mutant (both are fractions)
        (0.0, False, False),    # sole killer of a hardcoded gate
    ], ids=["just-below", "gate-off-while-broke"])
    def test_is_bankrupt_wiring(self, cash, enabled, expected):
        """The env threads its OWN cash/config into the shared rule.

        Deliberately goes through env._is_bankrupt(), not the shared helper: the helper's
        arithmetic is pinned in tests/envs/test_termination.py, and asserting it again here
        would pass even if this env swapped current/initial or hardcoded the gate.

        The boundary and the gate were both unpinned before: the old test drives cash to 0
        against a threshold of 5, far past the boundary.
        """
        env, _, _ = _make_env(
            config_overrides={
                "initial_cash": 1000.0,
                "done_on_bankruptcy": enabled,
                # 0.2, NOT the fixture's bet_fraction of 0.1 -- with both at 0.1 this test
                # cannot tell the two config fields apart, and passing the wrong one kills
                # nothing. -> a threshold of 200.0
                "bankrupt_threshold": 0.2,
            },
        )
        env.reset()
        env.cash = cash

        assert env._is_bankrupt() is expected

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
        """Termination only fires after env-level retries are exhausted."""
        attempts = 3
        env, scanner, _ = _make_env(
            outcomes=[1],
            markets=[_make_market()] + [None] * attempts,
            config_overrides={"next_market_max_attempts": attempts},
        )
        env.reset()
        reset_calls = scanner.next_active_market.call_count
        td = env._step(TensorDict({"action": torch.tensor(1)}, batch_size=()))
        assert td["terminated"].item()
        assert torch.equal(td["market_state"], torch.zeros(4))
        assert scanner.next_active_market.call_count - reset_calls == attempts

    def test_recovers_when_scanner_returns_none_then_market(self):
        """Transient None from scanner triggers retry, not termination."""
        env, scanner, _ = _make_env(
            outcomes=[1],
            markets=[_make_market(), None, _make_market(slug="btc-updown-5m-recovered")],
            config_overrides={"next_market_max_attempts": 3},
        )
        env.reset()
        reset_calls = scanner.next_active_market.call_count
        td = env._step(TensorDict({"action": torch.tensor(1)}, batch_size=()))
        assert not td["terminated"].item()
        assert not torch.equal(td["market_state"], torch.zeros(4))
        assert scanner.next_active_market.call_count - reset_calls == 2

    def test_retry_sleep_honors_configured_value_and_skips_after_final(self):
        """Pin the backoff contract: configured sleep value reaches time.sleep,
        and there is no extra sleep after the final failed attempt. Catches a
        regression that hardcodes the sleep or moves it outside the loop guard
        — neither shows up in tests with sleep defaulted to 0."""
        attempts = 3
        sleep_seconds = 7.5  # non-default to expose a hardcoded value
        env, _, _ = _make_env(
            outcomes=[1],
            markets=[_make_market()] + [None] * attempts,
            config_overrides={
                "next_market_max_attempts": attempts,
                "next_market_retry_sleep_seconds": sleep_seconds,
            },
        )
        env.reset()
        with patch("torchtrade.envs.live.polymarket.env.time.sleep") as mock_sleep:
            env._step(TensorDict({"action": torch.tensor(1)}, batch_size=()))
        assert mock_sleep.call_args_list == [call(sleep_seconds)] * (attempts - 1)

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
            ("0.99", "0.01", 1),     # EXACTLY at threshold (Up): >= must not become >
            ("0.01", "0.99", 0),     # EXACTLY at threshold (Down): <= must not become <
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
        # 7.0, NOT the 15.0 default: pinning the default cannot distinguish a config read
        # from a hardcoded constant. (The sibling backoff test at test_retry_sleep_... does
        # this correctly; I missed it here and pinned 15.0, which killed nothing.)
        env = self._real_poll_env(resolution_poll_interval_seconds=7.0)
        # First two calls: still mid-market. Third: Up wins.
        env._fetch_resolved_outcome = MagicMock(side_effect=[None, None, 1])
        with patch("torchtrade.envs.live.polymarket.env.time.sleep") as mock_sleep:
            assert env._poll_for_resolution("0xcond") == 1
        assert env._fetch_resolved_outcome.call_count == 3
        # Pin the poll interval BY VALUE, not just "it slept". Nothing pinned this, so
        # hardcoding a constant passed the whole suite -- turning a 15s poll into a 1s one,
        # i.e. ~600 polls per resolution instead of ~40 (and each poll is 2 CLOB requests).
        assert mock_sleep.call_args_list == [call(7.0), call(7.0)]

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
            # By VALUE, not "> 100": a 120s horizon plus the configured 30s grace. The loose
            # bound let `target = end_dt` (grace dropped entirely) pass, since 120 > 100.
            assert slept_for == pytest.approx(150, abs=2)


class TestConfigValidation:
    """The config is the boundary. A nonsense value must be refused there, not absorbed."""

    def test_live_mode_is_refused(self):
        """dry_run=False must RAISE, not silently paper-trade.

        The env can never execute live: py-clob-client is archived/non-functional, and the
        env holds every bet to resolution while Polymarket only releases collateral on an
        explicit on-chain redeem that no client exposes -- so a live bot's balance drains to
        zero while it is winning. Refusing the config is the whole point of this guard; a
        regression that quietly downgraded live to paper (or dropped the check) would let a
        user believe real money was at work.
        """
        # Match the SUBSTANCE, not a contentless phrase. Pinning only "not supported" let the
        # whole message be replaced with "pip install py-clob-client, then set dry_run=False"
        # -- the exact dead-end advice this PR exists to delete -- with the suite still green.
        # Both reasons must survive: the dead client AND the redemption blocker.
        with pytest.raises(NotImplementedError) as exc:
            PolymarketBetEnvConfig(market_slug_prefix="btc-updown-5m-", dry_run=False)
        assert "archived" in str(exc.value)   # blocker 1: the client is dead
        assert "redeem" in str(exc.value)     # blocker 2: the one a port cannot skip

    def test_config_is_frozen_so_the_guard_cannot_be_mutated_away(self):
        """__post_init__ runs ONCE. On a mutable dataclass this is the bypass:

            config = PolymarketBetEnvConfig(...)   # dry_run=True, passes
            config.dry_run = False                 # guard never re-runs
            env = PolymarketBetEnv(config)         # wires a LIVE trader

        frozen=True does not make the bypass impossible (object.__setattr__ still works);
        it makes it deliberate rather than an ordinary attribute assignment.
        """
        config = PolymarketBetEnvConfig(market_slug_prefix="btc-updown-5m-")
        with pytest.raises(dataclasses.FrozenInstanceError):
            config.dry_run = False

    @pytest.mark.parametrize("bad", [-1.0, 1.5], ids=["negative-disables-the-stop", "over-100%"])
    def test_bankrupt_threshold_validated_at_the_boundary(self, bad):
        """The same silent-default trap as initial_cash, one field over.

        > 1: threshold * initial exceeds the starting cash, so step 1 terminates unless the
        opening bet alone lifts cash back above the line.
        < 0: `current < threshold * initial` is unsatisfiable for any non-negative cash, which
        SILENTLY DISABLES the safety stop -- the exact failure this repo keeps getting bitten
        by, and the reason _is_bankrupt's old `initial > 0` guard was removed rather than kept.
        """
        with pytest.raises(ValueError, match="bankrupt_threshold"):
            PolymarketBetEnvConfig(market_slug_prefix="btc-updown-5m-", bankrupt_threshold=bad)

    @pytest.mark.parametrize("bad", [-0.1, 1.5], ids=["negative", "over-100%"])
    def test_bet_fraction_validated_at_the_boundary(self, bad):
        """bet_fraction > 1 stakes more than the account holds, driving cash negative -- which
        flips the sign of a loss, so a losing bet would PAY OUT.

        Refusing it here is what guarantees cash >= 0, and that guarantee is in turn why
        _compute_payoff carries no `stake <= 0` guard (it was deleted as unreachable). The
        boundary check REPLACES the downstream guard; it does not merely back it up.
        """
        with pytest.raises(ValueError, match="bet_fraction"):
            PolymarketBetEnvConfig(market_slug_prefix="btc-updown-5m-", bet_fraction=bad)

    def test_default_config_is_paper(self):
        """The DEFAULT must be the safe mode.

        Not a tautology against the line above: this pins that a user who never mentions
        dry_run gets a constructible paper env. The default used to be dry_run=False, which
        (once the guard exists) means the default config would not even build.
        """
        config = PolymarketBetEnvConfig(market_slug_prefix="btc-updown-5m-")
        assert config.dry_run is True

    @pytest.mark.parametrize("initial_cash", [-100.0, 0.0], ids=["negative", "zero"])
    def test_nonsense_starting_cash_is_refused(self, initial_cash):
        """Negative starting cash used to be silently absorbed into 'never bankrupt'.

        The old _is_bankrupt() carried an `initial > 0` guard, so a config with
        initial_cash=-100 constructed fine and simply never terminated. That is a nonsense
        config producing a silently disabled safety stop -- the exact silent-default pattern
        this repo keeps getting bitten by.
        """
        with pytest.raises(ValueError, match="initial_cash"):
            PolymarketBetEnvConfig(initial_cash=initial_cash)

    def test_a_sane_config_still_constructs(self):
        assert PolymarketBetEnvConfig(initial_cash=1000.0).initial_cash == 1000.0
