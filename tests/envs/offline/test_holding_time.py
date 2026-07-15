"""holding_time (account_state[3]) is the SAME across every offline env — issue #49.

The canonical rule (locked; see advance_hold_counter in core/state.py, which the live
envs share):

    flat            -> 0
    the OPENING bar -> 1          <-- offline used to report 0 here; that was #49
    each held bar   -> 2, 3, ...
    on close        -> 0
    a direct flip   -> 1          (new position, never passed through flat)

This file is the regression guard for the OFFLINE half. The live half is already pinned by
tests/envs/base_exchange_tests.py::assert_a_direct_flip_does_not_age_the_new_position (a fresh
position reads 1, and ages past 1) plus the test_reset_clears_the_holding_time... test in each live exchange.
The vectorized offline envs are covered transitively by test_vec_scalar_equivalence.py /
test_vec_sltp_scalar_equivalence.py, which already assert vec holding_time == scalar
holding_time step-for-step — so pinning the SCALAR sequence here fixes their expected value too.
"""

import pytest
import torch

from torchtrade.envs.offline import (
    SequentialTradingEnv,
    SequentialTradingEnvConfig,
    SequentialTradingEnvSLTP,
    SequentialTradingEnvSLTPConfig,
)
from torchtrade.envs.utils.timeframe import TimeFrame, TimeFrameUnit
from tests.conftest import simple_feature_fn


def _drive(env, actions):
    """Step `env` through `actions`, returning [reset holding_time, *per-step holding_time]."""
    td = env.reset()
    seq = [td["account_state"][3].item()]
    for a in actions:
        td["action"] = torch.tensor(a)
        td = env.step(td)["next"]
        seq.append(td["account_state"][3].item())
    return seq


# action_levels index maps: spot [0,1] -> {flat:0, long:1}; futures [-1,0,1] -> {short:0, flat:1, long:2}
@pytest.mark.parametrize("leverage,action_levels,actions,expected", [
    # spot: open long, hold, hold, close
    (1, [0, 1], [1, 1, 1, 0], [0, 1, 2, 3, 0]),
    # futures: open long, hold, hold, close, open short (opposite), hold
    (10, [-1, 0, 1], [2, 2, 2, 1, 0, 0], [0, 1, 2, 3, 0, 1, 2]),
    # futures DIRECT flip: open long, hold, flip straight to short (no flat bar), hold
    (10, [-1, 0, 1], [2, 2, 0, 0], [0, 1, 2, 1, 2]),
])
def test_sequential_holding_time_sequence(sample_ohlcv_df, leverage, action_levels, actions, expected):
    """The opening bar reads 1, holds increment, close resets to 0, a reopen restarts at 1."""
    config = SequentialTradingEnvConfig(
        leverage=leverage,
        action_levels=action_levels,
        initial_cash=1000,
        execute_on=TimeFrame(1, TimeFrameUnit.Minute),
        time_frames=[TimeFrame(1, TimeFrameUnit.Minute)],
        window_sizes=[10],
        transaction_fee=0.0,
        slippage=0.0,
        random_start=False,
    )
    env = SequentialTradingEnv(sample_ohlcv_df, config, simple_feature_fn)
    assert _drive(env, actions) == expected
    env.close()


@pytest.mark.parametrize("leverage", [1, 10], ids=["spot", "futures"])
def test_sltp_holding_time_sequence(sample_ohlcv_df, leverage):
    """SLTP env: opening a bracket reads 1 (was 0 pre-#49), then holds increment.

    Wide SL/TP (-2% / +3%) against the ~0.1%/bar synthetic data guarantees the bracket does
    NOT exit over these three bars, so the sequence is deterministic.
    """
    config = SequentialTradingEnvSLTPConfig(
        leverage=leverage,
        initial_cash=1000,
        execute_on=TimeFrame(1, TimeFrameUnit.Minute),
        time_frames=[TimeFrame(1, TimeFrameUnit.Minute)],
        window_sizes=[10],
        transaction_fee=0.0,
        slippage=0.0,
        random_start=False,
        stoploss_levels=[-0.02],
        takeprofit_levels=[0.03],
    )
    env = SequentialTradingEnvSLTP(sample_ohlcv_df, config, simple_feature_fn)
    # action 1 = open the (only) SL/TP bracket; action 0 = hold
    seq = _drive(env, [1, 0, 0])
    assert seq == [0, 1, 2, 3]
    env.close()
