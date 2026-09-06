"""Drive the INSTALLED torchtrade, with no checkout on sys.path.

Both tests.yml and release.yml run this. They had a copy each and the copies had
already drifted in the commit that created them: the release copy dropped the binance
import, so the check guarding the release was the weaker of the two.

Run it as a file path from a working directory outside the repo. Python puts the
script's own directory on sys.path, not the current one, so `import torchtrade`
resolves to site-packages, and the assert below fails loudly if it ever does not.
"""

import numpy as np
import pandas as pd
import torchtrade
from importlib.metadata import version

assert "site-packages" in torchtrade.__file__, (
    f"imported the checkout, not the installed package: {torchtrade.__file__}"
)
print("torchtrade", version("torchtrade"), "from", torchtrade.__file__)

from torchtrade.envs.offline.sequential import (
    SequentialTradingEnv,
    SequentialTradingEnvConfig,
)

n = 3000
close = 100 + np.cumsum(np.random.randn(n) * 0.1)
df = pd.DataFrame({
    "timestamp": pd.date_range("2024-01-01", periods=n, freq="1min", tz="UTC"),
    "open": close, "high": close + 0.5, "low": close - 0.5,
    "close": close, "volume": np.random.rand(n) * 10,
})
env = SequentialTradingEnv(df=df, config=SequentialTradingEnvConfig(
    time_frames=["1Min"], window_sizes=[10], execute_on="1Min", symbol="BTC/USD"))
td = env.reset()
td["action"] = env.action_spec.rand()
env.step(td)

from torchtrade.envs.live.alpaca import AlpacaTorchTradingEnv  # noqa: F401
from torchtrade.envs.live.binance import BinanceFuturesTorchTradingEnv  # noqa: F401
from torchtrade.envs.replay import ReplayObserver  # noqa: F401

print("offline env stepped; live and replay imports resolved")
