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
    ([0.5, 0.5, 0.5], [1 / 3] * 3),
    ([2.0, 0.0, 0.0], [1.0, 0.0, 0.0]),
    ([0.2, 0.3, 0.5], [0.2, 0.3, 0.5]),
    ([[0.5, 0.5, 0.5], [-1.0, 0.0, 1.0]], [[1 / 3] * 3, [0.0, 0.0, 1.0]]),
], ids=["shrink", "corner", "already-on-simplex", "batched"])
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
    assert all(t > 0 for t in env.history.turnovers[2:])  # drift is undone every step


def _olmar_td(window_closes, tradable, weights):
    """One observation: `window_closes` is (N, W) close over the latest close."""
    closes = torch.tensor(window_closes)
    market = torch.stack([closes, closes, closes], -1)
    return TensorDict({
        "market_data_1Hour_4": market, "tradable": torch.tensor(tradable), "portfolio_weights": torch.tensor(weights),
    })


def test_olmar_moves_toward_the_mean_reversion_prediction():
    """Two assets: A fell (mean/latest = 1.25, predicted to revert up), B rose (0.8). From an
    equal-weight book with epsilon 1.1, OLMAR-1 gives lambda = (1.1 - 1.0166) / 0.1017 and
    the projected update; B and cash lose weight to A."""
    td = _olmar_td([[1.5, 1.25, 1.25, 1.0], [0.6, 0.8, 0.8, 1.0]], [1.0, 1.0], [1 / 3, 1 / 3, 1 / 3])
    OLMAR(window=4, epsilon=1.1)(td)
    x_hat = torch.tensor([1.0, 1.25, 0.8])
    b = torch.full((3,), 1 / 3)
    centered = x_hat - x_hat.mean()
    lam = (1.1 - (b * x_hat).sum()) / (centered ** 2).sum()
    expected = project_simplex(b + lam * centered)
    torch.testing.assert_close(td["action"], expected)
    assert td["action"][1] > 1 / 3 > td["action"][2]


def test_olmar_gives_no_weight_to_untradable_assets_and_no_move_without_signal():
    """An unlisted asset (zero window) and a closed one get weight 0; a tradable asset with
    no bars yet is not read as a crash; a flat window keeps the book."""
    td = _olmar_td([[0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0], [1.3, 1.3, 1.3, 1.0]], [0.0, 1.0, 0.0], [0.5, 0.0, 0.5, 0.0])
    OLMAR(window=4, epsilon=1.1)(td)
    assert td["action"][1] == 0 and td["action"][3] == 0 and td["action"].sum() == pytest.approx(1.0)
    fresh = _olmar_td([[0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]], [1.0, 1.0], [0.2, 0.3, 0.5])
    OLMAR(window=4, epsilon=1.1)(fresh)
    torch.testing.assert_close(fresh["action"], torch.tensor([0.2, 0.3, 0.5]))
    flat = _olmar_td([[1.0] * 4, [1.0] * 4], [1.0, 1.0], [0.2, 0.3, 0.5])
    OLMAR(window=4, epsilon=1.1)(flat)
    torch.testing.assert_close(flat["action"], torch.tensor([0.2, 0.3, 0.5]))
