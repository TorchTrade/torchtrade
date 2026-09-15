import math

import pandas as pd
import pytest
import torch
from tensordict import TensorDict
from torchrl.envs.utils import check_env_specs

from tests.conftest import make_portfolio_bars
from torchtrade.envs.offline.infrastructure.portfolio_sampler import PortfolioSampler
from torchtrade.envs.offline.portfolio import PortfolioTradingEnv, PortfolioTradingEnvConfig
from torchtrade.envs.offline.vectorized_portfolio import (
    VectorizedPortfolioTradingEnv,
    VectorizedPortfolioTradingEnvConfig,
)

SMALL = dict(time_frames="1Hour", window_sizes=8, execute_on="4Hour", initial_cash=1000, random_start=False)


def _price_jump_bars(jump_ratio):
    """Bars where A1's price jumps by `jump_ratio` starting at the second execution bar.

    Deterministic (make_portfolio_bars is seeded): `jump_ratio` was picked by reading the
    actual realized close_exec ratio for A1, not guessed against the bars' natural drift.
    """
    bars = make_portfolio_bars()
    config = PortfolioTradingEnvConfig(**SMALL)
    probe = PortfolioSampler(bars, config.time_frames, config.window_sizes, config.execute_on, seed=config.seed)
    asset_idx = probe.inst_ids.index("A1")
    jump_ts = probe.exec_times[1]
    mask = (bars["inst_id"] == "A1") & (bars["timestamp"] >= jump_ts)
    bars.loc[mask, ["open", "high", "low", "close"]] *= jump_ratio
    return bars, asset_idx


def _delisted_bars():
    """make_portfolio_bars with A2's rows stopping 200 hours into the timeline."""
    bars = make_portfolio_bars()
    cutoff = bars["timestamp"].min() + pd.Timedelta(hours=200)
    return bars[~((bars["inst_id"] == "A2") & (bars["timestamp"] >= cutoff))].reset_index(drop=True)


@pytest.mark.parametrize("kwargs", [
    {"transaction_fee": -0.1}, {"transaction_fee": 1.0},
    {"max_gross": 0.0}, {"max_gross": 1.5}, {"bankrupt_threshold": 1.0},
    {"max_traj_length": 0},
], ids=lambda k: next(iter(k.items())).__repr__())
@pytest.mark.parametrize("config_cls", [PortfolioTradingEnvConfig, VectorizedPortfolioTradingEnvConfig],
                         ids=["scalar", "vectorized"])
def test_config_rejects_out_of_range(config_cls, kwargs):
    with pytest.raises(ValueError):
        config_cls(**kwargs)


@pytest.mark.parametrize("allow_short", [False, True])
@pytest.mark.parametrize("random_start", [False, True])
def test_check_env_specs(allow_short, random_start):
    config = PortfolioTradingEnvConfig(**{**SMALL, "random_start": random_start}, allow_short=allow_short)
    env = PortfolioTradingEnv(make_portfolio_bars(), config)
    check_env_specs(env)


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
@pytest.mark.parametrize("vectorized", [False, True])
def test_step_rejects_non_finite_action(vectorized, value):
    """A diverged policy emitting NaN/inf must fail loudly, not get silently absorbed."""
    bars = make_portfolio_bars()
    if vectorized:
        env = VectorizedPortfolioTradingEnv(bars, VectorizedPortfolioTradingEnvConfig(**SMALL, num_envs=2))
        env.reset()
        action = torch.zeros(2, 4)
        action[1, 1] = value  # lane 1 only: a lane-0-only check would miss this
        td = TensorDict({"action": action}, batch_size=[2])
    else:
        env = PortfolioTradingEnv(bars, PortfolioTradingEnvConfig(**SMALL))
        td = env.reset()
        td["action"] = torch.tensor([0.0, value, 0.0, 0.0])
    with pytest.raises(ValueError, match="non-finite"):
        env.step(td)


@pytest.mark.parametrize("fee,rate", [(0.0, 0.0), (0.001, 0.0), (0.001, 0.0005)])
def test_buy_and_hold_single_asset_end_to_end(fee, rate):
    """All in A0 once, then ask for exactly the drifted weights: only the entry fee and
    funding may separate the final value from the price ratio."""
    bars = make_portfolio_bars()
    funding = pd.DataFrame({
        "timestamp": pd.date_range("2026-01-05", "2026-01-19", freq="8h"),
        "inst_id": "A0", "funding_rate": rate,
    })
    env = PortfolioTradingEnv(bars, PortfolioTradingEnvConfig(**SMALL, transaction_fee=fee), funding=funding)
    td = env.reset()
    start, end = env._idx, env._end
    action = torch.tensor([0.0, 1.0, 0.0, 0.0])
    total_reward = 0.0
    while True:
        td["action"] = action
        td = env.step(td)["next"]
        total_reward += td["reward"].item()
        action = td["portfolio_weights"].clone()
        if td["done"].item():
            break
    s = env.sampler
    settlements = int((s.funding_exec[start:end, 0] != 0).sum())
    expected = 1000 / (1 + fee) * (s.close_exec[end, 0] / s.close_exec[start, 0]).item() * (1 - rate) ** settlements
    assert env.portfolio_value == pytest.approx(expected, rel=1e-6)
    assert total_reward == pytest.approx(math.log(env.portfolio_value / 1000), abs=1e-5)

    h = env.history.to_dict()
    assert {len(v) for v in h.values()} == {end - start + 1}
    assert h["timestamps"][-1] == s.exec_times[env._end]
    assert h["portfolio_values"][-1] == env.portfolio_value
    # Row 0 is the reset row; the entry trade pays V0·(1 − μ) with μ = 1/(1 + fee).
    assert h["commissions"][1] == pytest.approx(1000 * fee / (1 + fee), abs=1e-9)
    assert sum(h["commissions"][2:]) == pytest.approx(0.0, abs=1e-6)
    if rate == 0:
        assert all(f == 0.0 for f in h["fundings"])
    else:
        assert sum(f > 0 for f in h["fundings"]) == settlements


def test_observation_timing_matches_next_index():
    """The observation after step k must be the sampler's state at start + k, not at
    start + k - 1: fill at close n, value/observe at close n + 1. A0 is untradable on
    weekends so the flag genuinely changes mid-episode, and a stale index (reading bar n
    instead of n + 1) would fail both the tradable and market_data comparisons below."""
    bars = make_portfolio_bars()
    bars["tradable"] = True
    weekend = pd.to_datetime(bars["timestamp"]).dt.weekday >= 5
    bars.loc[(bars["inst_id"] == "A0") & weekend, "tradable"] = False

    env = PortfolioTradingEnv(bars, PortfolioTradingEnvConfig(**SMALL))
    td = env.reset()
    start, s = env._idx, env.sampler
    action = torch.tensor([1.0, 0.0, 0.0, 0.0])
    prev_tradable = td["tradable"].clone()
    seen_diff = False
    k = 0
    while True:
        td["action"] = action
        td = env.step(td)["next"]
        k += 1
        expected_idx = start + k
        assert torch.equal(td["tradable"], s.tradable_exec[expected_idx].float())
        expected_market = s.market_data(torch.tensor([expected_idx]))
        for key, _ in s.market_data_keys:
            assert torch.equal(td[key], expected_market[key][0])
        if not torch.equal(td["tradable"], prev_tradable):
            seen_diff = True
        prev_tradable = td["tradable"].clone()
        if td["done"].item():
            break
    assert seen_diff


@pytest.mark.parametrize("jump_ratio,expect_wiped", [(1.9, False), (2.05, True)], ids=["below-threshold", "wiped"])
def test_termination_on_price_jump(jump_ratio, expect_wiped):
    """An all-in short on A1, met by a large price jump between the first two execution
    bars, must terminate: below bankrupt_threshold if the short survives with a residual
    (growth in (0, bankrupt_threshold)), or wiped to exactly 0 if the jump exceeds 2x
    (growth <= 0, per portfolio_math's wiped-lane clamp).
    """
    bars, asset_idx = _price_jump_bars(jump_ratio)
    config = PortfolioTradingEnvConfig(**SMALL, allow_short=True)

    env = PortfolioTradingEnv(bars, config)
    td = env.reset()
    assert env.portfolio_value == pytest.approx(env.initial_portfolio_value)  # nothing terminates before the jump

    price_relative = (env.sampler.close_exec[1, asset_idx] / env.sampler.close_exec[0, asset_idx]).item()
    expected_growth = 2 - price_relative  # fee=0, cash=0, single -1 short weight on A1

    action = torch.zeros(len(env.inst_ids) + 1)
    action[asset_idx + 1] = -1.0
    td["action"] = action
    td = env.step(td)["next"]

    assert td["terminated"].item()
    assert td["done"].item()
    assert torch.isfinite(td["reward"]).all()
    torch.testing.assert_close(
        torch.tensor(env.history.weights[-1], dtype=torch.float32), td["portfolio_weights"], rtol=0, atol=1e-6
    )
    if expect_wiped:
        assert expected_growth <= 0  # sanity: the chosen jump really wipes the lane
        assert env.portfolio_value == 0.0
        assert td["reward"].item() == pytest.approx(-10.0)
        assert torch.isfinite(td["portfolio_weights"]).all()
    else:
        assert 0 < expected_growth < config.bankrupt_threshold  # sanity: below threshold, not wiped
        assert env.portfolio_value == pytest.approx(env.initial_portfolio_value * expected_growth)
        assert 0 < env.portfolio_value < config.bankrupt_threshold * env.initial_portfolio_value


@pytest.mark.parametrize("allow_short,weight", [(False, 1.0), (True, -1.0)], ids=["long", "short"])
def test_delist_force_close(allow_short, weight):
    """Asking for 100% of an asset every step must still be force-closed the step its
    data runs out, and stay closed (all cash, flat value) for every step after."""
    env = PortfolioTradingEnv(_delisted_bars(), PortfolioTradingEnvConfig(**SMALL, allow_short=allow_short))
    td = env.reset()
    asset_idx = env.inst_ids.index("A2")
    delist_n = int(env.sampler.delist_exec[asset_idx])
    assert delist_n >= 0  # sanity: the asset was actually delisted within the episode

    action = torch.zeros(len(env.inst_ids) + 1)
    action[asset_idx + 1] = weight
    values_after_close = []
    while True:
        n_before = env._idx
        td["action"] = action
        td = env.step(td)["next"]
        if n_before == delist_n:
            assert env.history.weights[-1][asset_idx + 1] == pytest.approx(0.0)
        if env._idx > delist_n:
            assert td["portfolio_weights"][asset_idx + 1].item() == pytest.approx(0.0)
            values_after_close.append(env.portfolio_value)
        if td["done"].item():
            break

    assert len(values_after_close) > 1
    assert all(v == pytest.approx(values_after_close[0]) for v in values_after_close)


def test_vectorized_config_rejects_zero_envs():
    with pytest.raises(ValueError):
        VectorizedPortfolioTradingEnvConfig(num_envs=0)


@pytest.mark.parametrize("allow_short", [False, True])
@pytest.mark.parametrize("random_start", [False, True])
def test_vectorized_check_env_specs(allow_short, random_start):
    env = VectorizedPortfolioTradingEnv(
        make_portfolio_bars(),
        VectorizedPortfolioTradingEnvConfig(**{**SMALL, "random_start": random_start}, num_envs=3, allow_short=allow_short),
    )
    check_env_specs(env)


def _random_action_scenario(allow_short):
    """Random actions each step, with closures, delisting, funding and fees mixed in."""
    bars = make_portfolio_bars(n_assets=4)
    closed = (bars.inst_id == "A1") & (bars.timestamp.dt.dayofweek >= 5)
    delisted = (bars.inst_id == "A3") & (bars.timestamp >= "2026-01-15")
    bars = bars[~closed & ~delisted]
    funding = pd.DataFrame([
        {"timestamp": ts, "inst_id": inst, "funding_rate": 0.0003 * (1 if i % 2 else -1)}
        for ts in pd.date_range("2026-01-05", "2026-01-19", freq="8h")
        for i, inst in enumerate(["A0", "A1", "A2", "A3"])
    ])
    return bars, funding, dict(**SMALL, transaction_fee=0.001, allow_short=allow_short), None


def _jump_scenario(jump_ratio, **overrides):
    bars, _ = _price_jump_bars(jump_ratio)
    return bars, None, dict(**SMALL, allow_short=True, **overrides), ("A1", -1.0)


def _delist_scenario():
    """100% into an asset whose rows stop mid-timeline, as in test_delist_force_close."""
    return _delisted_bars(), None, dict(**SMALL), ("A2", 1.0)


@pytest.mark.parametrize("scenario", [
    pytest.param(lambda: _random_action_scenario(False), id="random-long-only"),
    pytest.param(lambda: _random_action_scenario(True), id="random-short"),
    pytest.param(lambda: _jump_scenario(1.9), id="below-threshold"),
    pytest.param(lambda: _jump_scenario(2.05), id="wiped"),
    pytest.param(lambda: _jump_scenario(2.05, bankrupt_threshold=0.0), id="wiped-zero-threshold"),
    pytest.param(_delist_scenario, id="delisted"),
])
def test_scalar_and_vectorized_agree(scenario):
    """Same actions, same data: same everything, including the failure paths (per-lane
    termination, wipe-out and delisting), not just random actions that never terminate."""
    bars, funding, cfg, fixed = scenario()

    scalar = PortfolioTradingEnv(bars, PortfolioTradingEnvConfig(**cfg), funding=funding)
    vec = VectorizedPortfolioTradingEnv(bars, VectorizedPortfolioTradingEnvConfig(**cfg, num_envs=1), funding=funding)

    gen = torch.Generator().manual_seed(1)
    n = len(scalar.inst_ids)
    fixed_action = None
    if fixed is not None:
        name, weight = fixed
        fixed_action = torch.zeros(n + 1)
        fixed_action[scalar.inst_ids.index(name) + 1] = weight
    s_td, v_td = scalar.reset(), vec.reset()
    for _ in range(scalar._end - scalar._idx):
        action = torch.randn(n + 1, generator=gen) if fixed_action is None else fixed_action
        s_td["action"], v_td["action"] = action, action[None]
        s_td, v_td = scalar.step(s_td)["next"], vec.step(v_td)["next"]
        assert vec._pvs[0].item() == pytest.approx(scalar.portfolio_value, rel=1e-9, abs=0)
        for key in s_td.keys():
            torch.testing.assert_close(v_td[key][0], s_td[key], rtol=0, atol=1e-6)
        if s_td["done"].item():
            break
    assert s_td["done"].item()


def test_vectorized_per_lane_termination():
    """Two lanes stepped together, one pushed below the bankrupt threshold by a price jump and
    the other flat: only that lane terminates, and the flat lane's value is unaffected."""
    bars, asset_idx = _price_jump_bars(1.9)
    vec = VectorizedPortfolioTradingEnv(
        bars, VectorizedPortfolioTradingEnvConfig(**{**SMALL, "allow_short": True}, num_envs=2)
    )
    vec.reset()
    action = torch.zeros(2, 4)
    action[0, asset_idx + 1] = -1.0  # lane 0: all-in short on A1
    action[1, 0] = 1.0  # lane 1: all cash
    lane1_before = vec._pvs[1].item()

    td = TensorDict({"action": action}, batch_size=[2])
    td = vec.step(td)["next"]

    assert td["terminated"].tolist() == [[True], [False]]
    assert vec._pvs[1].item() == pytest.approx(lane1_before)


def test_vectorized_step_past_end_no_raise():
    """`_idx` is set directly because a rollout resets done lanes before stepping them again."""
    env = VectorizedPortfolioTradingEnv(make_portfolio_bars(), VectorizedPortfolioTradingEnvConfig(**SMALL, num_envs=2))
    env.reset()
    env._idx = torch.full_like(env._idx, env.sampler.num_exec - 1)
    td = TensorDict({"action": torch.tensor([[1.0, 0.0, 0.0, 0.0]] * 2)}, batch_size=[2])
    env.step(td)


def test_partial_reset_leaves_other_lanes_untouched():
    env = VectorizedPortfolioTradingEnv(make_portfolio_bars(), VectorizedPortfolioTradingEnvConfig(**SMALL, num_envs=3))
    td = env.reset()
    for _ in range(3):
        td["action"] = torch.tensor([[0.0, 1.0, 0.0, 0.0]] * 3)
        td = env.step(td)["next"]
    before = (env._pvs.clone(), env._drifted.clone(), env._idx.clone())
    held = td["portfolio_weights"]
    held_values = held.clone()
    env.reset(TensorDict({"_reset": torch.tensor([[False], [True], [False]])}, batch_size=[3]))
    # _reset writes the lane in place; an observation already handed out must not change.
    torch.testing.assert_close(held, held_values, rtol=0, atol=0)
    keep = torch.tensor([0, 2])
    for now, then in zip((env._pvs, env._drifted, env._idx), before):
        torch.testing.assert_close(now[keep], then[keep])
    assert env._pvs[1].item() == 1000 and env._idx[1].item() == 0
    assert env._drifted[1].tolist() == [1.0, 0.0, 0.0, 0.0]


@pytest.mark.parametrize("vectorized", [False, True], ids=["scalar", "vectorized"])
@pytest.mark.parametrize("max_traj_length,initial_cash", [
    pytest.param(None, 1000, id="to-end"),
    pytest.param(10, 1000, id="max-10"),
    pytest.param(None, (500, 1500), id="tuple-cash"),
])
def test_random_start_episode_windows(vectorized, max_traj_length, initial_cash):
    """Starts spread over the timeline, each episode ends at min(start + max_traj_length, last)."""
    cfg = {**SMALL, "random_start": True, "max_traj_length": max_traj_length, "initial_cash": initial_cash}
    action = torch.tensor([1.0, 0.0, 0.0, 0.0])
    if vectorized:
        env = VectorizedPortfolioTradingEnv(make_portfolio_bars(), VectorizedPortfolioTradingEnvConfig(**cfg, num_envs=64))
        env.set_seed(0)
        td = env.reset()
        starts, ends, cash = env._idx.clone(), env._end.clone(), env._initial_pvs.clone()
        pvs = env._pvs.clone()
        env.set_seed(0)
        env.reset()
        assert torch.equal(env._idx, starts) and torch.equal(env._end, ends) and torch.equal(env._pvs, pvs)
        env.set_seed(1)
        env.reset()
        assert not torch.equal(env._idx, starts)
        env.set_seed(0)
        td = env.reset()
        assert torch.equal(td["reset_index"], starts) and torch.equal(td["state_index"], starts)
        td["action"] = action.expand(64, -1)
        td = env.step(td)["next"]
        assert torch.equal(td["reset_index"], starts) and torch.equal(td["state_index"], env._idx)
    else:
        env = PortfolioTradingEnv(make_portfolio_bars(), PortfolioTradingEnvConfig(**cfg))
        env.set_seed(0)
        starts, ends, cash = [], [], []
        for _ in range(20):
            td = env.reset()
            start = env._idx
            assert td["reset_index"].item() == start and td["state_index"].item() == start
            starts.append(start)
            ends.append(env._end)
            cash.append(env.initial_portfolio_value)
            td["action"] = action
            td = env.step(td)["next"]
            assert td["reset_index"].item() == start and td["state_index"].item() == env._idx

        def windows(seed):
            env.set_seed(seed)
            pairs = []
            for _ in range(20):
                env.reset()
                pairs.append((env._idx, env._end))
            return pairs

        assert windows(0) == list(zip(starts, ends)) and windows(1) != list(zip(starts, ends))
        starts, ends, cash = torch.tensor(starts), torch.tensor(ends), torch.tensor(cash)

    last = env.sampler.num_exec - 1
    steps = last - starts if max_traj_length is None else torch.full_like(starts, max_traj_length)
    assert len(starts.unique()) > 1
    assert torch.equal(ends - starts, torch.minimum(steps, last - starts))
    assert (ends <= last).all()
    if isinstance(initial_cash, tuple):
        assert ((cash >= 500) & (cash <= 1500)).all()


@pytest.mark.parametrize("make_env", [
    pytest.param(
        lambda bars: PortfolioTradingEnv(bars, PortfolioTradingEnvConfig(**SMALL), reward_function=lambda history: 1.0),
        id="scalar",
    ),
    pytest.param(
        lambda bars: VectorizedPortfolioTradingEnv(
            bars, VectorizedPortfolioTradingEnvConfig(**SMALL, num_envs=2),
            reward_function=lambda old, new: torch.ones_like(new),
        ),
        id="vectorized",
    ),
])
def test_reward_function_injection(make_env):
    rollout = make_env(make_portfolio_bars()).rollout(5)
    assert (rollout["next", "reward"] == 1.0).all()
