"""Fixtures shared by every venue's test directory (#288).

`replay_df` lived in seven places: six venue test classes defined it inline and okx's
conftest held a seventh copy. They were the same 20 lines, and the two that had drifted
had drifted in the RNG stream rather than in anything a reader would notice.
"""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def replay_df():
    """Realistic OHLCV for the replay integration tests.

    Drawn in this order so the RNG stream is what the original fixtures produced, then
    clamped: a close drawn off `base` can land outside a high/low drawn off `base` alone,
    which is a bar the venue would never print (#326).
    """
    n = 200
    rng = np.random.default_rng(42)
    base = 50000 + np.cumsum(rng.normal(0, 50, n))
    high_raw = base + np.abs(rng.normal(30, 20, n))
    low_raw = base - np.abs(rng.normal(30, 20, n))
    close = base + rng.normal(0, 20, n)
    return pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=n, freq="1min"),
        "open": base,
        "high": np.maximum(high_raw, np.maximum(base, close)),
        "low": np.minimum(low_raw, np.minimum(base, close)),
        "close": close,
        "volume": rng.uniform(100, 1000, n),
    })
