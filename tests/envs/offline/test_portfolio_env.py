import math

import pandas as pd
import pytest
import torch
from torchrl.envs.utils import check_env_specs

from tests.conftest import make_portfolio_bars
from torchtrade.envs.offline.infrastructure.portfolio_sampler import PortfolioSampler
from torchtrade.envs.offline.portfolio import PortfolioTradingEnv, PortfolioTradingEnvConfig

SMALL = dict(time_frames="1Hour", window_sizes=8, execute_on="4Hour", initial_cash=1000, random_start=False)


@pytest.mark.parametrize("kwargs", [
    {"transaction_fee": -0.1}, {"transaction_fee": 1.0},
    {"max_gross": 0.0}, {"max_gross": 1.5}, {"bankrupt_threshold": 1.0},
], ids=lambda k: next(iter(k.items())).__repr__())
def test_config_rejects_out_of_range(kwargs):
    with pytest.raises(ValueError):
        PortfolioTradingEnvConfig(**kwargs)


@pytest.mark.parametrize("allow_short", [False, True])
@pytest.mark.parametrize("random_start", [False, True])
def test_check_env_specs(allow_short, random_start):
    config = PortfolioTradingEnvConfig(**{**SMALL, "random_start": random_start}, allow_short=allow_short)
    env = PortfolioTradingEnv(make_portfolio_bars(), config)
    check_env_specs(env)


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
    (growth <= 0, per portfolio_math's wiped-lane clamp). `jump_ratio` was picked by
    reading the actual realized close_exec ratio for A1 (make_portfolio_bars is seeded,
    so this is deterministic, not a guess against the natural drift already in the bars).
    """
    bars = make_portfolio_bars()
    config = PortfolioTradingEnvConfig(**SMALL, allow_short=True)
    probe = PortfolioSampler(bars, config.time_frames, config.window_sizes, config.execute_on, seed=config.seed)
    asset_idx = probe.inst_ids.index("A1")
    jump_ts = probe.exec_times[1]
    mask = (bars["inst_id"] == "A1") & (bars["timestamp"] >= jump_ts)
    bars.loc[mask, ["open", "high", "low", "close"]] *= jump_ratio

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
    if expect_wiped:
        assert expected_growth <= 0  # sanity: the chosen jump really wipes the lane
        assert env.portfolio_value == 0.0
        assert td["reward"].item() == pytest.approx(-10.0)
        assert torch.isfinite(td["portfolio_weights"]).all()
    else:
        assert 0 < expected_growth < config.bankrupt_threshold  # sanity: below threshold, not wiped
        assert env.portfolio_value == pytest.approx(env.initial_portfolio_value * expected_growth)
        assert 0 < env.portfolio_value < config.bankrupt_threshold * env.initial_portfolio_value


def test_delist_force_close():
    """Asking for 100% of an asset every step must still be force-closed the step its
    data runs out, and stay closed (all cash, flat value) for every step after."""
    bars = make_portfolio_bars()
    cutoff = bars["timestamp"].min() + pd.Timedelta(hours=200)
    bars = bars[~((bars["inst_id"] == "A2") & (bars["timestamp"] >= cutoff))].reset_index(drop=True)

    env = PortfolioTradingEnv(bars, PortfolioTradingEnvConfig(**SMALL))
    td = env.reset()
    asset_idx = env.inst_ids.index("A2")
    delist_n = int(env.sampler.delist_exec[asset_idx])
    assert delist_n >= 0  # sanity: the asset was actually delisted within the episode

    action = torch.zeros(len(env.inst_ids) + 1)
    action[asset_idx + 1] = 1.0
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
