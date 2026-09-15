import numpy as np
import pandas as pd
import pytest
import torch

from tests.conftest import make_portfolio_bars
from torchtrade.envs.offline.infrastructure.portfolio_sampler import PortfolioSampler
from torchtrade.envs.utils.timeframe import normalize_timeframe_config


def _sampler(bars, time_frames=("1Hour",), window_sizes=(8,), execute_on="4Hour", funding=None):
    ex, tfs, ws = normalize_timeframe_config(execute_on, list(time_frames), list(window_sizes))
    return PortfolioSampler(bars, tfs, ws, ex, funding=funding, seed=0)


def _bad_bars(kind):
    bars = make_portfolio_bars()
    if kind == "missing-column":
        return bars.drop(columns="volume")
    if kind == "duplicate-row":
        return pd.concat([bars, bars.iloc[:1]])
    bars.loc[5, "close"] = np.nan
    return bars


@pytest.mark.parametrize("kind,match", [
    ("missing-column", "missing required columns"),
    ("duplicate-row", "duplicate"),
    ("nan-price", "NaN"),
])
def test_invalid_bars_raise(kind, match):
    with pytest.raises(ValueError, match=match):
        _sampler(_bad_bars(kind))


@pytest.mark.parametrize("funding,match", [
    (pd.DataFrame({"timestamp": [pd.Timestamp("2026-01-06")], "inst_id": ["ZZZ"], "funding_rate": [0.001]}), "unknown inst_id"),
    (pd.DataFrame({"timestamp": [pd.Timestamp("2026-01-06")] * 2, "inst_id": ["A0"] * 2, "funding_rate": [0.001] * 2}), "duplicate"),
])
def test_invalid_funding_raises(funding, match):
    with pytest.raises(ValueError, match=match):
        _sampler(make_portfolio_bars(), funding=funding)


def test_weekend_closure_carries_price_and_blocks_trading():
    bars = make_portfolio_bars()
    weekend = (bars.inst_id == "A1") & (bars.timestamp.dt.dayofweek >= 5)
    s = _sampler(bars[~weekend])
    a1 = s.inst_ids.index("A1")
    is_weekend = torch.from_numpy(s.exec_times.dayofweek.to_numpy() >= 5)
    assert not s.tradable_exec[is_weekend, a1].any()
    assert s.tradable_exec[~is_weekend].all()
    friday_close = bars[(bars.inst_id == "A1") & ~weekend & (bars.timestamp < "2026-01-10")].close.iloc[-1]
    assert torch.all(s.close_exec[is_weekend, a1][:12] == friday_close)


@pytest.mark.parametrize("time_frames,window_sizes,hours,listed_from", [
    pytest.param(("1Hour",), (8,), 24 * 14, "2026-01-08 06:00", id="fine"),
    pytest.param(("1Day", "1Hour"), (3, 8), 24 * 20, "2026-01-12 06:00", id="coarse-first-key"),
])
def test_listing_zeroes_features_and_blocks_trading_before_the_first_row(
    time_frames, window_sizes, hours, listed_from
):
    bars = make_portfolio_bars(hours=hours)
    late = (bars.inst_id == "A2") & (bars.timestamp < listed_from)
    s = _sampler(bars[~late], time_frames=time_frames, window_sizes=window_sizes)
    a2 = s.inst_ids.index("A2")
    # A2 lists mid-bin (06:00, inside the [04:00, 08:00) execute_on bin), so the bin
    # itself is only PARTLY unlisted -- exclude it from the all-zero check below and
    # pin it on its own: tradable_exec must read the bin's LAST base bar (07:00, listed),
    # not resample's default first (04:00, still unlisted).
    partial_bin = pd.Timestamp(listed_from).floor("4h")
    before = torch.from_numpy(np.flatnonzero(s.exec_times < partial_bin))
    key = s.market_data_keys[0][0]
    assert torch.all(s.market_data(before)[key][:, a2] == 0)
    assert not s.tradable_exec[before, a2].any()

    mid_bin = int(np.flatnonzero(s.exec_times == partial_bin)[0])
    empty_bin = int(np.flatnonzero(s.exec_times == partial_bin - pd.Timedelta("4h"))[0])
    assert s.tradable_exec[mid_bin, a2]
    assert not s.tradable_exec[empty_bin, a2]


@pytest.mark.parametrize("drop_after,expected_delisted", [(None, False), ("2026-01-12", True)])
def test_delisting_is_rows_ending_before_the_timeline(drop_after, expected_delisted):
    bars = make_portfolio_bars()
    if drop_after:
        bars = bars[~((bars.inst_id == "A0") & (bars.timestamp >= drop_after))]
    s = _sampler(bars)
    a0 = s.inst_ids.index("A0")
    if not expected_delisted:
        assert s.delist_exec[a0] == -1
    else:
        last = int(s.delist_exec[a0])
        assert s.tradable_exec[last, a0] and not s.tradable_exec[last + 1:, a0].any()


def test_coarse_frame_flags_are_end_labelled():
    """A day bar is visible only once it has closed: the listing day must still read unlisted."""
    bars = make_portfolio_bars(hours=24 * 20)
    bars = bars[~((bars.inst_id == "A2") & (bars.timestamp < "2026-01-12 06:00"))]
    s = _sampler(bars, time_frames=("1Hour", "1Day"), window_sizes=(8, 3))
    a2 = s.inst_ids.index("A2")
    idx = torch.tensor([int(np.flatnonzero(s.exec_times == "2026-01-12 08:00")[0])])
    day_window = s.market_data(idx)["market_data_1Day_3"][0, a2]
    assert torch.all(day_window[-1] == 0)


def _funding_at(s0, n, delta, rate=0.001):
    """One funding row settling `delta` after window n's fill (exec_times[n] + 4h)."""
    ts = s0.exec_times[n] + pd.Timedelta("4h") + delta
    return pd.DataFrame({"timestamp": [ts], "inst_id": ["A0"], "funding_rate": [rate]})


def _funding_two_in_same_step(s0, n):
    """Two funding rows both settling inside window n's (fill_n, fill_{n+1}] span."""
    fill_n = s0.exec_times[n] + pd.Timedelta("4h")
    return pd.DataFrame({
        "timestamp": [fill_n + pd.Timedelta("1h"), fill_n + pd.Timedelta("2h")],
        "inst_id": ["A0", "A0"],
        "funding_rate": [0.001, 0.001],
    })


@pytest.mark.parametrize("build_funding,expected", [
    pytest.param(lambda s0: _funding_at(s0, 5, pd.Timedelta(0)), {4: 0.001}, id="fill"),
    pytest.param(lambda s0: _funding_at(s0, 5, pd.Timedelta("1h")), {5: 0.001}, id="inside"),
    pytest.param(lambda s0: _funding_at(s0, 5, pd.Timedelta("4h")), {5: 0.001}, id="next-fill"),
    pytest.param(lambda s0: _funding_at(s0, -1, pd.Timedelta("4h") + pd.Timedelta("1h")), {}, id="after-last-window"),
    pytest.param(lambda s0: _funding_at(s0, -1, pd.Timedelta("4h")), {"last": 0.001}, id="at-last-window-edge"),
    pytest.param(lambda s0: _funding_two_in_same_step(s0, 5), {5: 0.002}, id="same-step-sums"),
])
def test_funding_window(build_funding, expected):
    s0 = _sampler(make_portfolio_bars())
    funding = build_funding(s0)
    s = _sampler(make_portfolio_bars(), funding=funding)
    resolved = {(s.num_exec - 1 if k == "last" else k): v for k, v in expected.items()}
    charged = {i: float(s.funding_exec[i, 0]) for i in torch.nonzero(s.funding_exec[:, 0]).flatten().tolist()}
    assert charged == resolved
