import pytest
import torch
from tensordict import TensorDict

from tests.conftest import make_portfolio_bars
from torchtrade.actor import OLMAR, UBAH, UCRP
from torchtrade.actor.rulebased.portfolio import project_simplex
from torchtrade.envs.offline import (
    PortfolioTradingEnv,
    PortfolioTradingEnvConfig,
    VectorizedPortfolioTradingEnv,
    VectorizedPortfolioTradingEnvConfig,
)

SMALL = dict(time_frames="1Hour", window_sizes=8, execute_on="4Hour", initial_cash=1000, random_start=False)


def _env(vectorized, **kwargs):
    cfg = {**SMALL, **kwargs}
    if vectorized:
        return VectorizedPortfolioTradingEnv(make_portfolio_bars(), VectorizedPortfolioTradingEnvConfig(**cfg, num_envs=2))
    return PortfolioTradingEnv(make_portfolio_bars(), PortfolioTradingEnvConfig(**cfg))


@pytest.mark.parametrize("v,expected", [
    ([2.0, 0.0, 0.0], [1.0, 0.0, 0.0]),
    ([3.0, 1.0, 0.0], [1.0, 0.0, 0.0]),  # clip-and-renormalise would give [0.75, 0.25, 0]
    ([[0.5, 0.5, 0.5], [-1.0, 0.0, 1.0]], [[1 / 3] * 3, [0.0, 0.0, 1.0]]),
    ([3e7, 0.0], [1.0, 0.0]),  # float32 loses the unit offset at this magnitude
], ids=["corner", "truncation", "batched", "float32-magnitude"])
def test_project_simplex(v, expected):
    torch.testing.assert_close(project_simplex(torch.tensor(v)), torch.tensor(expected))


@pytest.mark.parametrize("vectorized", [False, True], ids=["scalar", "vectorized"])
@pytest.mark.parametrize("baseline", [UBAH(), UCRP(), OLMAR(window=4)], ids=["UBAH", "UCRP", "OLMAR"])
def test_baselines_roll_out_on_both_envs(vectorized, baseline):
    """A plain callable is a policy: every action is on the simplex and inside the spec."""
    env = _env(vectorized)
    td = env.rollout(5, policy=baseline)
    action = td["action"]
    assert env.action_spec.is_in(action[..., 0, :])
    torch.testing.assert_close(action.sum(-1), torch.ones_like(action.sum(-1)))
    assert (action >= 0).all()


def test_ubah_buys_equal_weights_once_then_holds():
    env = _env(False, transaction_fee=0.001)
    env.rollout(6, policy=UBAH())
    h = env.history.to_dict()
    assert h["turnovers"][1] == pytest.approx(1.0) and h["commissions"][1] > 0
    assert sum(h["turnovers"][2:]) == pytest.approx(0.0, abs=1e-6)  # float32 observation round trip
    torch.testing.assert_close(torch.tensor(h["weights"][1][1:]), torch.full((3,), 1 / 3), atol=0.05, rtol=0)


def test_ucrp_rebalances_to_the_same_target_every_step():
    env = _env(False)
    td = env.rollout(4, policy=UCRP())
    torch.testing.assert_close(td["action"], torch.tensor([0.0, 1 / 3, 1 / 3, 1 / 3]).expand(4, -1))
    # Undoing drift moves weight between assets with zero net change: a signed turnover would read 0.
    assert all(t > 0 for t in env.history.turnovers[2:])


def _olmar_td(window_closes, tradable, weights):
    """One observation with a 6-bar window: `window_closes` is (N, 6) close over the latest close.
    High and low differ from close, and a decoy second market-data key holds the reversed window."""
    closes = torch.tensor(window_closes)
    market = torch.stack([closes, closes * 1.02, closes * 0.98], -1)
    return TensorDict({
        "market_data_1Hour_6": market, "market_data_1Day_6": market.flip(-2),
        "tradable": torch.tensor(tradable), "portfolio_weights": torch.tensor(weights),
    })


A_FELL = [0.9, 0.9, 1.5, 1.25, 1.25, 1.0]  # last 4 bars: mean over latest = 1.25, predicted to revert up
B_ROSE = [1.4, 1.4, 0.6, 0.8, 0.8, 1.0]  # last 4 bars: 0.8


def test_olmar_moves_toward_the_mean_reversion_prediction():
    """From an equal-weight book with epsilon 1.1, OLMAR-1 gives lambda = (1.1 - 1.0167) / 0.1017
    on the last `window` bars of the first market-data key; B and cash lose weight to A."""
    td = _olmar_td([A_FELL, B_ROSE], [1.0, 1.0], [1 / 3, 1 / 3, 1 / 3])
    OLMAR(window=4, epsilon=1.1)(td)
    x_hat = torch.tensor([1.0, 1.25, 0.8])
    b = torch.full((3,), 1 / 3)
    centered = x_hat - x_hat.mean()
    lam = (1.1 - (b * x_hat).sum()) / (centered ** 2).sum()
    torch.testing.assert_close(td["action"], project_simplex(b + lam * centered))
    assert td["action"][1] > 1 / 3 > td["action"][2]


FLAT = [1.0] * 6


@pytest.mark.parametrize("windows,tradable,weights,epsilon,expected", [
    pytest.param([[0.0] * 6, FLAT, [1.3] * 5 + [1.0]], [0.0, 1.0, 0.0], [0.5, 0.0, 0.0, 0.5], 1.1, [0.5, 0.0, 0.0, 0.5],
                 id="untradable-assets-give-no-signal"),
    pytest.param([FLAT, FLAT], [1.0, 1.0], [0.0, -0.5, -0.5], 1.1, [1 / 3, 1 / 3, 1 / 3], id="all-short-book-starts-from-nothing"),
    pytest.param([[0.0] * 6, FLAT], [1.0, 1.0], [0.2, 0.3, 0.5], 1.1, [0.2, 0.3, 0.5], id="tradable-without-bars-is-no-signal"),
    pytest.param([FLAT, FLAT], [1.0, 1.0], [0.2, 0.3, 0.5], 1.1, [0.2, 0.3, 0.5], id="flat-window-keeps-the-book"),
    pytest.param([A_FELL, B_ROSE], [1.0, 1.0], [0.0, 1.0, 0.0], 1.1, [0.0, 1.0, 0.0], id="prediction-above-epsilon-no-move"),
    pytest.param([[0.0, 0.0, 0.0, 0.0, 1.2, 1.0]], [1.0], [1.0, 0.0], 1.05, [0.5, 0.5], id="mid-window-listing-counts-its-own-bars"),
    pytest.param([FLAT, FLAT], [1.0, 1.0], [0.5, 0.7, -0.2], 1.1, [5 / 12, 7 / 12, 0.0], id="short-book-clipped-and-renormalised"),
    # B is closed: its bars are no signal, so b.x_hat = 1.125 sits above epsilon and nothing moves.
    pytest.param([A_FELL, B_ROSE], [1.0, 0.0], [0.0, 0.5, 0.5], 1.1, [0.0, 0.5, 0.5], id="closed-asset-bars-are-no-signal"),
    # A 0.03% signal still moves all in: the denominator floor is a division guard, not damping.
    pytest.param([[1.0004] * 5 + [1.0]], [1.0], [1.0, 0.0], 1.1, [0.0, 1.0], id="tiny-signal-still-moves-all-in"),
])
def test_olmar_edge_cases(windows, tradable, weights, epsilon, expected):
    td = _olmar_td(windows, tradable, weights)
    OLMAR(window=4, epsilon=epsilon)(td)
    torch.testing.assert_close(td["action"], torch.tensor(expected))
