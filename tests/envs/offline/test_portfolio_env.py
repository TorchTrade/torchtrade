import math

import pandas as pd
import pytest
import torch
from torchrl.envs.utils import check_env_specs

from tests.conftest import make_portfolio_bars
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
