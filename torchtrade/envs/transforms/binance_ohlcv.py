"""TorchRL transform that augments observations with multi-timeframe Binance OHLCV.

Generic side-channel data injector, wrap any TorchRL env in
``TransformedEnv(env, BinanceOHLCVTransform(...))`` and the observation gains
one tensor per ``(timeframe, window)`` pair, fetched live from Binance's public
klines endpoint each step. No API key required (Binance allows unauthenticated
read-only access to public market data).

Designed for the Polymarket short-cadence betting use case (BTC up/down 5 min
markets benefit from live BTC price action), but works as a drop-in augment
for any env where crypto OHLCV is signal-relevant. Swap the
``feature_preprocessing_fn`` to inject your own technicals / microfeatures.
"""

from __future__ import annotations

import logging
from typing import Callable, List, Optional

import numpy as np
import torch
from tensordict import TensorDictBase
from torchrl.data import Bounded
from torchrl.envs.transforms import Transform

from torchtrade.envs.live.binance.observation import BinanceObservationClass
from torchtrade.envs.utils.timeframe import TimeFrame, TimeFrameUnit

logger = logging.getLogger(__name__)


class BinanceOHLCVTransform(Transform):
    """Append multi-timeframe Binance OHLCV windows to an env's observation.

    Each step (and on reset) the transform calls
    :meth:`BinanceObservationClass.get_observations` and writes one tensor per
    ``(timeframe, window)`` pair into the next-step TensorDict, keyed
    ``{key_prefix}_{TimeFrame.obs_key_freq()}_{window}``. The
    ``observation_spec`` is extended with matching ``Bounded`` entries so
    ``check_env_specs`` passes.

    Args:
        symbol: Binance symbol (default ``"BTCUSDT"``). Leading slashes are
            stripped automatically (so ``"BTC/USDT"`` works).
        time_frames: List of :class:`TimeFrame` objects. Defaults to 1m / 5m /
            15m, sensible for a 5-minute polymarket betting env.
        window_sizes: List matching ``time_frames`` length. Defaults to
            60 / 30 / 20.
        feature_preprocessing_fn: Optional ``df -> df`` function that adds
            feature columns (any column whose name contains ``"feature"``).
            Defaults to Binance's normalized OHLC pct-change features.
        observer: Optional pre-built :class:`BinanceObservationClass` for
            dependency injection (used by tests). When provided, the other
            data-source kwargs are ignored.
        key_prefix: Prefix for the keys written into the observation
            TensorDict. Default ``"ohlcv"`` produces keys like
            ``ohlcv_5Minute_30``.

    Example:
        >>> from torchrl.envs import TransformedEnv
        >>> from torchtrade.envs.live.polymarket import (
        ...     PolymarketBetEnv, PolymarketBetEnvConfig,
        ... )
        >>> from torchtrade.envs.transforms import BinanceOHLCVTransform
        >>> env = TransformedEnv(
        ...     PolymarketBetEnv(PolymarketBetEnvConfig(
        ...         market_slug_prefix="btc-updown-5m-", dry_run=True,
        ...     )),
        ...     BinanceOHLCVTransform(symbol="BTCUSDT"),
        ... )
        >>> td = env.reset()
        >>> sorted(k for k in td.keys() if k.startswith("ohlcv_"))  # doctest: +SKIP
        ['ohlcv_15Minute_20', 'ohlcv_1Minute_60', 'ohlcv_5Minute_30']
    """

    def __init__(
        self,
        symbol: str = "BTCUSDT",
        time_frames: Optional[List[TimeFrame]] = None,
        window_sizes: Optional[List[int]] = None,
        feature_preprocessing_fn: Optional[Callable] = None,
        observer: Optional[BinanceObservationClass] = None,
        key_prefix: str = "ohlcv",
    ):
        super().__init__(in_keys=[], out_keys=[])

        if observer is not None:
            self.observer = observer
        else:
            self.observer = BinanceObservationClass(
                symbol=symbol,
                time_frames=time_frames or [
                    TimeFrame(1, TimeFrameUnit.Minute),
                    TimeFrame(5, TimeFrameUnit.Minute),
                    TimeFrame(15, TimeFrameUnit.Minute),
                ],
                window_sizes=window_sizes or [60, 30, 20],
                feature_preprocessing_fn=feature_preprocessing_fn,
            )

        self._key_prefix = key_prefix
        # Discover the per-window feature width from the (cheap, CPU-only)
        # dummy preprocessing run rather than guessing it.
        self._n_features = len(
            self.observer.get_features()["observation_features"]
        )

    # ------------------------------------------------------------------ #
    #  Internal                                                           #
    # ------------------------------------------------------------------ #

    def _key(self, tf: TimeFrame, window: int) -> str:
        return f"{self._key_prefix}_{tf.obs_key_freq()}_{window}"

    def _attach_observations(self, td: TensorDictBase) -> TensorDictBase:
        """Set every declared key on the TensorDict.

        Every ``(timeframe, window)`` declared in ``transform_observation_spec``
        is also written here, if the observer returns a partial payload, the
        missing keys are filled with zeros and a warning is logged. Skipping
        them would let the runtime output drift from the spec, which would
        crash downstream collectors and policies that trust the spec.
        """
        obs = self.observer.get_observations()
        for tf, window in zip(
            self.observer.time_frames, self.observer.window_sizes, strict=True
        ):
            source_key = f"{tf.obs_key_freq()}_{window}"
            target_key = self._key(tf, window)
            if source_key in obs:
                value = torch.as_tensor(np.asarray(obs[source_key]), dtype=torch.float32)
            else:
                logger.warning(
                    "Observer omitted key %r, filling with zeros to honor observation_spec",
                    source_key,
                )
                value = torch.zeros((window, self._n_features), dtype=torch.float32)
            td.set(target_key, value)
        return td

    # ------------------------------------------------------------------ #
    #  TorchRL Transform hooks                                            #
    # ------------------------------------------------------------------ #

    def _reset(
        self, tensordict: TensorDictBase, tensordict_reset: TensorDictBase
    ) -> TensorDictBase:
        return self._attach_observations(tensordict_reset)

    def _step(
        self, tensordict: TensorDictBase, next_tensordict: TensorDictBase
    ) -> TensorDictBase:
        return self._attach_observations(next_tensordict)

    def transform_observation_spec(self, observation_spec):
        for tf, window in zip(
            self.observer.time_frames, self.observer.window_sizes, strict=True
        ):
            observation_spec.set(
                self._key(tf, window),
                Bounded(
                    low=-float("inf"),
                    high=float("inf"),
                    shape=(window, self._n_features),
                    dtype=torch.float32,
                ),
            )
        return observation_spec
