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
