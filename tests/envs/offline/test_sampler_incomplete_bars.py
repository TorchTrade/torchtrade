import numpy as np, pandas as pd, pytest
from torchtrade.envs.offline.infrastructure.sampler import MarketDataObservationSampler
from torchtrade.envs.utils.timeframe import TimeFrame, TimeFrameUnit


def _frame(n=400):
    idx = pd.date_range("2024-01-01", periods=n, freq="1min")
    return pd.DataFrame({"timestamp": idx, "open": 100.0, "high": 101.0,
                         "low": 99.0, "close": 100.0, "volume": 10.0})


@pytest.mark.parametrize("field", ["open", "high", "low"])
def test_an_incomplete_bar_does_not_inherit_the_previous_bar_s_price(field):
    """#353: gap detection was close-only, then the base was forward-filled wholesale.

    A bar with a good close and one missing price field was never warned about and
    silently carried the PREVIOUS bar's value -- so stop_fill_price could fill a gapped
    stop at a price the market never traded, and high/low answered "was this level
    touched?" with a range from one bar earlier.
    """
    df = _frame()
    df.loc[200, ["open", "high", "low", "close"]] = [200.0, 205.0, 195.0, 200.0]
    df.loc[201, field] = np.nan
    stale = df.loc[200, field]
    when = df.loc[201, "timestamp"]

    with pytest.warns(UserWarning, match="DATA GAPS"):
        sampler = MarketDataObservationSampler(
            df, time_frames=[TimeFrame(1, TimeFrameUnit.Minute)], window_sizes=[10],
            execute_on=TimeFrame(1, TimeFrameUnit.Minute),
        )

    bar = sampler.execute_base_features_df.loc[when]
    assert bar[field] != stale, f"{field} inherited the previous bar's price ({stale})"
    assert bar[field] == bar["close"], "an incomplete bar should collapse to its own close"


def test_a_bar_missing_only_its_open_keeps_its_real_high_and_low():
    """A row mask overwrote the fields that were PRESENT.

    Collapsing all three when one is missing turns a bar with a real 195-205 range into a
    flat bar on which no stop can trigger -- a backtest that skips losing exits, which is
    worse than the forward-fill it replaced. It was also a regression: ffill corrupted
    only `open` and left the real extremes alone.
    """
    df = _frame()
    df.loc[201, ["open", "high", "low", "close"]] = [np.nan, 205.0, 195.0, 200.0]
    when = df.loc[201, "timestamp"]

    with pytest.warns(UserWarning, match="DATA GAPS"):
        sampler = MarketDataObservationSampler(
            df, time_frames=[TimeFrame(1, TimeFrameUnit.Minute)], window_sizes=[10],
            execute_on=TimeFrame(1, TimeFrameUnit.Minute),
        )

    bar = sampler.execute_base_features_df.loc[when]
    assert bar["high"] == 205.0, "a real high was discarded because open was missing"
    assert bar["low"] == 195.0, "a real low was discarded because open was missing"
    assert bar["open"] == 200.0, "the missing open should take this bar's own close"


def test_a_stale_close_cannot_land_outside_its_own_bar():
    """`close` becomes current_price, the next entry, the mark and the reward.

    A bar with a real range and a NaN close was caught by neither collapse, so it took the
    PREVIOUS bar's close -- which can sit outside its own low/high, the exact malformed
    shape the ingestion validator rejects on raw input.
    """
    df = _frame()
    df.loc[200, ["open", "high", "low", "close"]] = [100.0, 101.0, 99.0, 100.0]
    df.loc[201, ["open", "high", "low", "close"]] = [200.0, 210.0, 190.0, np.nan]
    when = df.loc[201, "timestamp"]

    with pytest.warns(UserWarning, match="DATA GAPS"):
        sampler = MarketDataObservationSampler(
            df, time_frames=[TimeFrame(1, TimeFrameUnit.Minute)], window_sizes=[10],
            execute_on=TimeFrame(1, TimeFrameUnit.Minute),
        )

    bar = sampler.execute_base_features_df.loc[when]
    assert bar["low"] <= bar["close"] <= bar["high"], (
        f"close {bar['close']} sits outside its own bar [{bar['low']}, {bar['high']}]"
    )
