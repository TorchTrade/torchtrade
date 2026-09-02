"""Shared test bodies for the per-exchange live-env test files.

- BaseObservationClassTests: subclassed by each exchange's test_obs_class.py.
- assert_a_direct_flip_does_not_age_the_new_position: called by each exchange's
  test_torch_env_futures*.py, so the flip contract has one body and five kills.
"""


import pytest
import numpy as np
import torch
from tensordict import TensorDict
from unittest.mock import MagicMock, patch
from abc import ABC, abstractmethod
from torchtrade.envs.utils.timeframe import TimeFrame, TimeFrameUnit


def mirror_features_on(observer):
    """Derive a mock observer's get_features from the observations it actually emits.

    The envs build observation_spec from get_features() (#288), so a fixture that stubs
    only get_observations declares width 0 against emitted 4 -- which is exactly what the
    binance and bitget check_env_specs tests caught when the switch landed. Deriving
    means a fixture cannot declare a width its own data contradicts.
    """
    width = next(iter(observer.get_observations().values())).shape[1]
    observer.get_features = MagicMock(return_value={
        "observation_features": [f"feature_{i}" for i in range(width)],
        "original_features": [],
    })


class BaseObservationClassTests(ABC):
    """
    Base test class for exchange observation classes.

    Tests observation fetching, preprocessing, and feature extraction.
    Each exchange should subclass this and implement the abstract methods.
    """

    @abstractmethod
    def create_observer(self, symbol, timeframes, window_sizes, **kwargs):
        """
        Create an observer instance for the specific exchange.

        Args:
            symbol: Trading symbol
            timeframes: Single TimeFrame or list of TimeFrames
            window_sizes: Single int or list of ints
            **kwargs: Exchange-specific parameters

        Returns:
            Observer instance
        """
        pass

    @abstractmethod
    def get_expected_symbol_format(self, symbol):
        """
        Get the expected symbol format for this exchange.

        Args:
            symbol: Input symbol (e.g., "BTC/USD", "BTCUSDT")

        Returns:
            Expected normalized symbol format
        """
        pass

    # Initialization tests

    def test_init_single_timeframe(self):
        """Test initialization with single timeframe."""
        observer = self.create_observer(
            symbol="BTC/USD",
            timeframes=TimeFrame(15, TimeFrameUnit.Minute),
            window_sizes=20,
        )

        timeframes = getattr(observer, 'timeframes', getattr(observer, 'time_frames', None))
        window_sizes = observer.window_sizes

        assert len(timeframes) == 1
        assert timeframes[0].value == 15
        assert window_sizes[0] == 20

    def test_init_multiple_timeframes(self):
        """Test initialization with multiple timeframes."""
        observer = self.create_observer(
            symbol="BTC/USD",
            timeframes=[
                TimeFrame(1, TimeFrameUnit.Minute),
                TimeFrame(5, TimeFrameUnit.Minute),
                TimeFrame(1, TimeFrameUnit.Hour),
            ],
            window_sizes=[10, 20, 30],
        )

        timeframes = getattr(observer, 'timeframes', getattr(observer, 'time_frames', None))
        assert len(timeframes) == 3
        assert len(observer.window_sizes) == 3

    def test_init_mismatched_lengths_raises_error(self):
        """Test that mismatched timeframes and window_sizes raises ValueError."""
        with pytest.raises(ValueError, match="same length"):
            self.create_observer(
                symbol="BTC/USD",
                timeframes=[
                    TimeFrame(1, TimeFrameUnit.Minute),
                    TimeFrame(5, TimeFrameUnit.Minute),
                ],
                window_sizes=[10, 20, 30],  # 3 sizes for 2 timeframes
            )

    # get_keys tests

    def test_get_keys_single_timeframe(self):
        """Test get_keys with single timeframe."""
        observer = self.create_observer(
            symbol="BTC/USD",
            timeframes=TimeFrame(15, TimeFrameUnit.Minute),
            window_sizes=10,
        )

        keys = observer.get_keys()
        assert len(keys) == 1
        assert "15Minute_10" in keys[0] or "15m_10" in keys[0].lower()

    def test_get_keys_multiple_timeframes(self):
        """Test get_keys with multiple timeframes."""
        observer = self.create_observer(
            symbol="BTC/USD",
            timeframes=[
                TimeFrame(1, TimeFrameUnit.Minute),
                TimeFrame(1, TimeFrameUnit.Hour),
            ],
            window_sizes=[10, 20],
        )

        keys = observer.get_keys()
        assert len(keys) == 2

    # get_observations tests

    def test_get_observations_single_timeframe(self):
        """Test getting observations for single timeframe."""
        observer = self.create_observer(
            symbol="BTC/USD",
            timeframes=TimeFrame(1, TimeFrameUnit.Minute),
            window_sizes=10,
        )

        observations = observer.get_observations()

        assert isinstance(observations, dict)
        assert len(observations) >= 1  # At least one key

        # Check first observation
        key = observer.get_keys()[0]
        assert key in observations
        assert isinstance(observations[key], np.ndarray)
        assert observations[key].shape[0] == 10  # window_size
        assert observations[key].shape[1] >= 4  # At least 4 features (OHLC-based)

    def test_get_observations_multiple_timeframes(self):
        """Test getting observations for multiple timeframes."""
        observer = self.create_observer(
            symbol="BTC/USD",
            timeframes=[
                TimeFrame(1, TimeFrameUnit.Minute),
                TimeFrame(5, TimeFrameUnit.Minute),
            ],
            window_sizes=[10, 20],
        )

        observations = observer.get_observations()

        assert len(observations) >= 2
        keys = observer.get_keys()
        assert observations[keys[0]].shape == (10, 4) or observations[keys[0]].shape[0] == 10
        assert observations[keys[1]].shape == (20, 4) or observations[keys[1]].shape[0] == 20

    def test_get_observations_with_base_ohlc(self):
        """Test getting observations with base OHLC data."""
        observer = self.create_observer(
            symbol="BTC/USD",
            timeframes=TimeFrame(1, TimeFrameUnit.Minute),
            window_sizes=10,
        )

        observations = observer.get_observations(return_base_ohlc=True)

        assert "base_features" in observations
        assert observations["base_features"].shape[1] == 4  # OHLC


    def test_observations_are_float32(self):
        """Test that observations are float32."""
        observer = self.create_observer(
            symbol="BTC/USD",
            timeframes=TimeFrame(1, TimeFrameUnit.Minute),
            window_sizes=10,
        )

        observations = observer.get_observations()
        key = observer.get_keys()[0]

        assert observations[key].dtype == np.float32

    def test_observations_no_nan_values(self):
        """Test that observations don't contain NaN values."""
        observer = self.create_observer(
            symbol="BTC/USD",
            timeframes=TimeFrame(1, TimeFrameUnit.Minute),
            window_sizes=10,
        )

        observations = observer.get_observations()
        key = observer.get_keys()[0]

        assert not np.isnan(observations[key]).any()

    # get_features tests

    def test_the_declared_feature_width_matches_the_observation(self):
        """get_features() counts columns on a SYNTHETIC frame; get_observations() runs
        the same fn on real data. Nothing cross-checked that they agree.

        The fn must READ a dummy-supplied column, or this cannot fail: with default
        preprocessing the width is a constant 4 whatever the dummy contains, and the
        first version of this test passed even with `volume` deleted from the dummy
        frame outright. That width is what the venue transforms size their spec from.
        """
        def fn(df):
            df = df.copy()
            df["feature_range"] = (df["high"] - df["low"]) / df["close"]
            df["feature_vol"] = df["volume"] / df["volume"].mean()
            return df.dropna()

        observer = self.create_observer(
            symbol="BTC/USD",
            timeframes=TimeFrame(1, TimeFrameUnit.Minute),
            window_sizes=10,
            feature_preprocessing_fn=fn,
        )
        key = observer.get_keys()[0]
        declared = len(observer.get_features()["observation_features"])
        assert declared == observer.get_observations()[key].shape[1] == 2

    def test_get_features_default_preprocessing(self):
        """Test get_features with default preprocessing."""
        observer = self.create_observer(
            symbol="BTC/USD",
            timeframes=TimeFrame(1, TimeFrameUnit.Minute),
            window_sizes=10,
        )

        features = observer.get_features()

        assert "observation_features" in features
        assert "original_features" in features
        assert len(features["observation_features"]) >= 4  # At least OHLC features

    # Custom preprocessing tests

    def test_custom_preprocessing(self):
        """Test with custom preprocessing function."""
        def custom_preprocessing(df):
            df = df.copy()
            df.dropna(inplace=True)
            df["feature_volatility"] = df["high"] - df["low"]
            df["feature_volume_ma"] = df["volume"].rolling(window=3).mean()
            df.dropna(inplace=True)
            return df

        observer = self.create_observer(
            symbol="BTC/USD",
            timeframes=TimeFrame(1, TimeFrameUnit.Minute),
            window_sizes=10,
            feature_preprocessing_fn=custom_preprocessing,
        )

        observations = observer.get_observations()
        key = observer.get_keys()[0]

        # Custom preprocessing has 2 features
        assert observations[key].shape[1] == 2

    @pytest.mark.parametrize("kept,refused", [
        pytest.param(0, True, id="total-outage"),
        pytest.param(4, True, id="malformed-burst"),
        pytest.param(5, False, id="exactly-enough"),
    ])
    def test_an_observation_shorter_than_the_declared_spec_is_refused(self, kept, refused):
        """`iloc[-window_size:]` is a silent short read, not an error (#400).

        Preprocessing drops rows, so a burst of malformed candles exceeding the fetch
        buffer emitted a (4, n) array against a declared (5, n) spec, and reset() and
        rollout() both SUCCEEDED on it. Empty is the degenerate case, and it surfaced
        instead as an IndexError from `base_features[-1, 3]` inside the trade path
        (#397). Here, not per-venue: alpaca hand-rolls this while the four futures
        venues share one base, which is exactly how a fix lands on some and not others.
        """
        def truncating(df):
            df = df.copy()
            df["feature_range"] = df["high"] - df["low"]
            return df.iloc[:kept]

        observer = self.create_observer(
            symbol="BTC/USD",
            timeframes=TimeFrame(1, TimeFrameUnit.Minute),
            window_sizes=5,
            feature_preprocessing_fn=truncating,
        )
        if not refused:  # the boundary: `<` -> `<=` passes the whole suite without it
            emitted = observer.get_observations(return_base_ohlc=True)
            assert emitted["base_features"].shape[0] == 5
            return
        with pytest.raises(ValueError, match="usable candles"):
            observer.get_observations(return_base_ohlc=True)

    @pytest.mark.parametrize("timeframes,window_sizes", [
        (TimeFrame(1, TimeFrameUnit.Minute), [10, 20]),
        ([TimeFrame(1, TimeFrameUnit.Minute), TimeFrame(5, TimeFrameUnit.Minute)], 10),
    ], ids=["one-timeframe-two-windows", "two-timeframes-one-window"])
    def test_a_length_mismatch_raises_instead_of_truncating(self, timeframes, window_sizes):
        """`zip()` truncates, so a mismatch used to become a SHORTER observation set.

        Alpaca checked only when BOTH arguments were already lists, so passing one of
        each normalised to lists of different length and then silently lost a timeframe
        in `get_keys()` -- the policy trains on a window it was not configured for. The
        futures venues always checked unconditionally; folding onto the shared base is
        what made alpaca agree (#288).
        """
        with pytest.raises(ValueError, match="same length"):
            self.create_observer(symbol="BTC/USD", timeframes=timeframes,
                                 window_sizes=window_sizes)

    def test_the_shared_preprocessing_semantics_hold(self):
        """`_default_preprocessing` is one implementation now, so assert it on every venue.

        These three assertions lived only in alpaca's file. That was fine while alpaca
        owned its own copy; after the fold they cover code all five venues run, and a
        regression would have shipped silently for the four that had no equivalent (#288).
        """
        observer = self.create_observer(
            symbol="BTC/USD", timeframes=TimeFrame(1, TimeFrameUnit.Minute),
            window_sizes=10)
        obs = observer.get_observations(return_base_ohlc=True)

        # feature_close is a pct_change, so it sits near zero rather than near price.
        assert np.abs(obs[observer.get_keys()[0]][:, 0]).max() < 0.1

        # A column swap in the base_features slice would break high >= low.
        base = obs["base_features"]
        assert np.all(base[:, 1] >= base[:, 2])

        # base_timestamps is windowed alongside base_features; a mis-slice reorders it.
        timestamps = obs["base_timestamps"]
        assert len(timestamps) == len(base)
        assert np.all(timestamps[:-1] < timestamps[1:])

    def test_preprocessing_does_not_mutate_the_fetched_frame(self):
        """`_normalise_frame` returns a COPY, and the #399 guard depends on it.

        `_default_preprocessing` drops rows with `inplace=True`, and `get_observations`
        then compares the processed frame's last timestamp against the FETCHED frame's to
        refuse a stale bar. Share one object between them and both sides move together,
        so the comparison can never differ and a stale bar reaches the policy. Measured:
        returning `df` instead of `df.copy()` passes the entire suite.
        """
        observer = self.create_observer(
            symbol="BTC/USD", timeframes=TimeFrame(1, TimeFrameUnit.Minute),
            window_sizes=5)
        fetched = observer._fetch_single_timeframe(observer.time_frames[0], limit=55)
        before = (len(fetched), list(fetched.columns))

        observer.feature_preprocessing_fn(fetched)

        assert (len(fetched), list(fetched.columns)) == before, (
            "preprocessing mutated the frame it was handed, so the stale-bar check "
            "compares an object against itself"
        )

    def test_this_venue_does_not_re_fork_the_shared_window_logic(self):
        """One `get_observations`, for all five venues (#288).

        Alpaca kept a byte-for-byte parallel copy of the window logic until this landed,
        and each of the last three fixes to it -- row alignment (#395), the stale last bar
        (#399), the short window (#400) -- had to be pasted into both. A re-fork puts that
        back, and every behavioural test still passes on the day it happens, because both
        copies are correct at birth.

        `_dummy_frame` and `_normalise_frame` are extension points, not re-forks: venues
        return different columns, and their SDKs hand back different frame shapes.

        Name-based, deliberately: this diffs method NAMES and never reads a body, so a
        venue that reimplements the pipeline INSIDE one of those two exempted overrides
        passes. That is a copy-paste-drift guard, not a circumvention guard -- the
        behavioural tests above are what would catch the reimplementation.
        """
        from torchtrade.envs.live.shared.base_obs import BaseObservationClass

        venue_cls = type(self.create_observer(
            symbol="BTC/USD", timeframes=TimeFrame(1, TimeFrameUnit.Minute),
            window_sizes=5,
        ))
        shared = {
            n for n, v in vars(BaseObservationClass).items()
            if callable(v) and not getattr(v, "__isabstractmethod__", False)
            and not n.startswith("__")
        }
        assert {"get_observations", "_default_preprocessing", "get_features"} <= shared
        redeclared = (shared & set(vars(venue_cls))) - {"_dummy_frame", "_normalise_frame"}
        assert not redeclared, (
            f"{venue_cls.__name__} re-forks {sorted(redeclared)} instead of sharing "
            f"BaseObservationClass's"
        )

    # Edge cases

    def test_window_size_one(self):
        """Test with window size of 1."""
        observer = self.create_observer(
            symbol="BTC/USD",
            timeframes=TimeFrame(1, TimeFrameUnit.Minute),
            window_sizes=1,
        )

        observations = observer.get_observations()
        key = observer.get_keys()[0]

        assert observations[key].shape[0] == 1

    def test_large_window_size(self):
        """Test with large window size."""
        observer = self.create_observer(
            symbol="BTC/USD",
            timeframes=TimeFrame(1, TimeFrameUnit.Minute),
            window_sizes=100,
        )

        observations = observer.get_observations()
        key = observer.get_keys()[0]

        assert observations[key].shape[0] == 100


def assert_a_direct_flip_does_not_age_the_new_position(
    env, trader, PositionStatus, long_action, short_action
):
    """A long flipped straight to a short is ONE step old, not the long's age.

    THE TRAP: the exchange must still report the LONG when _step syncs -- the trade has not
    executed yet. Reporting the short there makes _sync_position_from_exchange see an EXTERNAL
    change and reset hold_counter itself (PR #245 / SLTPMixin), masking the bug: the test goes
    vacuous. Two of these passed on the buggy code before I found that.
    """
    # Keyword-built: since #289 every venue's PositionStatus has the same field names, so
    # the positional form this used to need (they differed only in field 8's spelling) is
    # gone. Only qty is read; the rest is inert filler.
    def pos(qty):
        liq = 45000.0 if qty > 0 else 55000.0     # shorts liquidate ABOVE the mark
        return {"position_status": PositionStatus(
            qty=qty, notional_value=500.0, entry_price=50000.0, unrealized_pnl=0.0,
            unrealized_pnl_pct=0.0, mark_price=50000.0, leverage=5,
            margin_mode="isolated", liquidation_price=liq)}

    with patch.object(env, "_wait_for_next_timestamp"):
        trader.get_status = MagicMock(return_value=pos(0.01))
        env.reset()

        for _ in range(5):
            td = env.step(TensorDict({"action": torch.tensor(long_action)}, batch_size=()))
        aged = td["next"]["account_state"][3].item()
        assert aged > 1.0, f"the long never aged ({aged}) -- the assertion below would be vacuous"


        # The exchange reports the OLD long until the trade actually executes -- keyed off the
        # TRADE, not off how many times _step happens to call get_status(). An earlier
        # get_status() added to _step (a price prefetch, a retry) would consume a
        # call-ordering-based mock's single "long", the sync would then see the short, reset
        # hold_counter ITSELF, and all five tests would go green on buggy code.
        traded = []
        inner_trade = trader.trade
        trader.trade = MagicMock(
            side_effect=lambda *a, **k: (traded.append(1), inner_trade(*a, **k))[1]
        )
        trader.get_status = MagicMock(side_effect=lambda: pos(-0.01) if traded else pos(0.01))
        td = env.step(TensorDict({"action": torch.tensor(short_action)}, batch_size=()))

    holding_time = td["next"]["account_state"][3].item()
    assert holding_time == 1.0, (
        f"a one-step-old short reports {holding_time} bars (the long aged to {aged}): the flip "
        f"never passed through flat, so the counter was never reset"
    )


def assert_the_step_emits_the_whole_done_family(env):
    """A stepped live env emits done, terminated AND truncated (#272).

    Deliberate coverage, replacing a tripwire that #313 disarmed. While every live _step
    wrote truncated by hand, a done spec missing the key showed up in check_env_specs as
    keys_in_real_not_in_fake. Once the hardcoded writes went, both the real and the fake
    rollout lack it, so check_env_specs cannot see a narrowed spec at all -- verified: the
    same spec mutation fails on the pre-#313 code and passes after it.

    So the spec is now the sole source of truncated, and this is what pins it. Goes
    through step(), not _step(): filling the family from the spec is EnvBase's job, and
    asserting it on _step's raw output would pin an implementation detail instead.
    """
    with patch.object(type(env), "_wait_for_next_timestamp"):
        td = env.reset()
        td["action"] = torch.tensor(0)
        nxt = env.step(td)["next"]

    for key in ("done", "terminated", "truncated"):
        assert key in nxt.keys(), (
            f"{key} missing from the emitted step. The live envs write the done family "
            "through the shared _finalize_step_flags (#295), so a key absent here means "
            "an env stopped routing through it."
        )
        assert nxt[key].dtype is torch.bool, f"{key} is {nxt[key].dtype}, not bool"
        assert nxt[key].shape == (1,), f"{key} has shape {tuple(nxt[key].shape)}, not (1,)"
    # Scoped to the DEFAULT config, which disables the grace period. Unscoped this reads
    # as "a live env never truncates", which #295 made false -- and it would keep passing
    # only because these fixtures leave max_unknown_status_steps at 0.
    budget = getattr(env.config, "max_unknown_status_steps", 0)   # absent on alpaca
    assert budget == 0 and not nxt["truncated"].any(), (
        "a live env does not truncate itself with the outage budget disabled"
    )


# The two inputs that pre-fix produced a SILENT TRADE rather than a crash, which is what
# a per-venue wiring regression would reintroduce. `-1` wrapped a list into a full long;
# `True` is subtler -- hash(True) == hash(1), so on the unguarded SLTP venues
# `action_map[True]` aliased `action_map[1]` and opened a real bracket order. The other
# malformed kinds (past-the-end, fractional, non-finite) already raised IndexError,
# TypeError or KeyError before any order, so at venue level they prove only that the
# exception type is now unified. The exhaustive sweep over all six kinds belongs to the
# validator itself, in tests/envs/test_live_env_base.py.
INVALID_ACTIONS = [
    pytest.param(torch.tensor(-1), id="negative"),
    pytest.param(torch.tensor(True), id="bool"),
]


def assert_an_invalid_action_raises_before_trading(env, action):
    """A malformed action index must raise, and must not move any money doing it.

    Every one of these was previously a *trade*, not an error -- `-1` wrapped a list into
    a full long, bybit/okx clamped it to a full short, `NaN` fell back to index 0. The
    load-bearing assertions are the last two: the buggy paths raised nothing at all.

    The venue-level callers exist because the original bug was per-venue WIRING -- alpaca
    never called the helper at all -- so a shared-contract test alone would not have
    caught it. They start FLAT, which makes the position assertion `== 0`; the open-
    position case, where a clamp is a full reversal, is
    `assert_an_invalid_action_cannot_move_an_open_position`.
    """
    from torchtrade.envs.core.live import InvalidActionError

    with patch.object(env, "_wait_for_next_timestamp"):
        env.reset()
        env.trader.trade.reset_mock()
        before = env.position.current_position
        with pytest.raises(InvalidActionError):
            env.step(TensorDict({"action": action}, batch_size=()))

    assert not env.trader.trade.called, (
        f"action {action!r} raised, but only after submitting "
        f"{env.trader.trade.call_args_list} -- the check must precede the order"
    )
    assert env.position.current_position == before, (
        f"action {action!r} left the position at {env.position.current_position}, "
        f"was {before}"
    )


def assert_an_invalid_action_cannot_move_an_open_position(env, action):
    """The expensive direction: an invalid action arriving while a position is OPEN.

    Every other invalid-action test starts flat, which makes the `position unchanged`
    assertion `== 0` and blind to the cases that cost the most. A regression that refuses
    invalid actions when flat and clamps them when a position is open passes the entire
    live-env suite while turning `-1` into a full reversal -- closing the long AND opening
    a short, roughly 2x notional through the book, from a malformed index.

    The sync is stubbed rather than the exchange mocked open, so this stays venue-
    agnostic: `_sync_position_from_exchange` runs BEFORE the resolve on 9 of 10 envs and
    would legitimately rewrite `current_position` back to flat from the MagicMock's
    default `{"position_status": None}`, which is a correct sync, not the regression.
    """
    from torchtrade.envs.core.live import InvalidActionError

    with patch.object(env, "_wait_for_next_timestamp"), \
         patch.object(env, "_sync_position_from_exchange", return_value=False):
        env.reset()
        env.position.current_position = 1
        env.trader.trade.reset_mock()
        with pytest.raises(InvalidActionError):
            env.step(TensorDict({"action": action}, batch_size=()))

    assert not env.trader.trade.called, (
        f"action {action!r} traded {env.trader.trade.call_args_list} against an OPEN "
        f"position -- on [-1, 0, 1] a clamp to index 0 is a full reversal, not a hold"
    )
    assert env.position.current_position == 1, (
        f"action {action!r} moved the open position to {env.position.current_position}"
    )


def wire_outage_state(env, budget=0):
    """Give a stub the #295 staleness attributes `__init__` would have set.

    Ten sites were pasting these four lines. A stub that skips them fails with an
    AttributeError from whichever read happens to touch them first, which says nothing
    about what the test was checking.
    """
    env.consecutive_unknown_status = 0
    env._status_unknown_this_step = False
    env._last_confirmed_read = {}
    env._max_unknown_status_steps = budget

def _sole(module, suffix):
    """The ONE class in `module` whose name ends in `suffix`.

    Not `next(...)`, which returns the first match in DEFINITION order: a decoy declared
    above the real class silently becomes the subject, and a guard then passes green on a
    class nobody meant to check.
    """
    found = [
        v for k, v in vars(module).items()
        if isinstance(v, type)          # `__module__` resolves through the class, so a
        and k.endswith(suffix)          # module-level INSTANCE would otherwise qualify
        and getattr(v, "__module__", None) == module.__name__
    ]
    assert len(found) == 1, (
        f"{module.__name__} defines {len(found)} classes ending in {suffix!r} "
        f"({[c.__name__ for c in found]}); the guard would have silently picked the first"
    )
    return found[0]


def _replay_env(env_cls, config_cls, replay_df, **config_kw):
    """A real venue env driven by ReplayObserver + ReplayOrderExecutor (#288).

    Eight venue test classes built this identically, differing only in the two classes
    and the config values -- which are exactly the two things this takes as arguments.
    """
    from torchtrade.envs.replay import ReplayObserver, ReplayOrderExecutor

    config = config_cls(
        **{"time_frames": ["1m"], "window_sizes": [10], "execute_on": "1m",
           "leverage": 5, **config_kw}   # merged, so a caller may override any of them
    )
    executor = ReplayOrderExecutor(initial_balance=10000.0, leverage=5)
    observer = ReplayObserver(
        df=replay_df,
        time_frames=config.time_frames,
        window_sizes=config.window_sizes,
        execute_on=config.execute_on,
        executor=executor,
    )
    with patch("time.sleep"), patch.object(env_cls, "_wait_for_next_timestamp"):
        env = env_cls(config=config, observer=observer, trader=executor)
    return env, executor


def assert_a_replay_episode_runs(env_cls, config_cls, replay_df, *, actions, steps,
                                 **config_kw):
    """Step a real episode over real price data and check the tensordict each bar.

    `actions` is a callable taking the bar index and the env, so the plain envs can cycle
    action LEVELS and the SLTP ones can cycle bracket indices -- the one thing that
    genuinely differed between the eight copies.
    """
    env, executor = _replay_env(env_cls, config_cls, replay_df, **config_kw)
    with patch.object(env, "_wait_for_next_timestamp"):
        td = env.reset()
        for i in range(steps):
            action_td = td.clone()
            action_td["action"] = torch.tensor(actions(i, env))
            td = env.step(action_td)["next"]

            assert "reward" in td.keys()
            assert "done" in td.keys()
            assert td["account_state"].shape == (6,)
            if td["done"].item():
                break

    assert executor.current_price > 0


def assert_the_replay_portfolio_tracks_price(env_cls, config_cls, replay_df, **config_kw):
    """Open a position, hold, and require the portfolio value to MOVE.

    A static value is the failure this catches: an env that never re-reads the mark
    reports the same equity every bar, which looks like a working episode.
    """
    env, executor = _replay_env(env_cls, config_cls, replay_df, **config_kw)
    with patch.object(env, "_wait_for_next_timestamp"):
        td = env.reset()
        action_td = td.clone()
        action_td["action"] = torch.tensor(1)          # open
        td = env.step(action_td)["next"]

        balances = []
        for _ in range(10):
            action_td = td.clone()
            action_td["action"] = torch.tensor(0)      # hold
            td = env.step(action_td)["next"]
            balances.append(executor.get_account_balance()["total_wallet_balance"])

    assert max(balances) != min(balances), (
        "portfolio value stayed static across ten bars of moving price"
    )
