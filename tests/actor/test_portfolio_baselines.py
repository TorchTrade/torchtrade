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


@pytest.mark.parametrize("max_gross,ok", [(1.0, True), (1e-9, True), (0.0, False), (1.5, False)], ids=["one", "tiny", "zero", "above-one"])
def test_ubah_max_gross_must_be_in_the_envs_range(max_gross, ok):
    if ok:
        assert UBAH(max_gross=max_gross).max_gross == max_gross
    else:
        with pytest.raises(ValueError, match="max_gross"):
            UBAH(max_gross=max_gross)


def _staggered_bars():
    """A2 cannot trade until 2026-01-06 12:00; the other two lanes trade from the start."""
    bars = make_portfolio_bars()
    bars["tradable"] = ((bars["inst_id"] != "A2") | (bars["timestamp"] >= "2026-01-06 12:00")).astype(int)
    return bars


@pytest.mark.parametrize("vectorized", [False, True], ids=["scalar", "vectorized"])
def test_ubah_buys_each_asset_when_it_first_trades(vectorized):
    """A lane closed at the first decision is bought at 1/N when it opens; the cash for it waits."""
    bars = _staggered_bars()
    if vectorized:
        env = VectorizedPortfolioTradingEnv(bars, VectorizedPortfolioTradingEnvConfig(**SMALL, num_envs=2))
    else:
        env = PortfolioTradingEnv(bars, PortfolioTradingEnvConfig(**SMALL))
    opened = int(env.sampler.tradable_exec[:, 2].nonzero()[0])  # the first decision at which A2 trades
    assert opened > 0, "the fixture must start with A2 closed"
    td = env.rollout(opened + 4, policy=UBAH())
    books = td["next", "portfolio_weights"]  # the drifted book after each decision: (T, N+1), or (B, T, N+1) vectorized
    third = torch.tensor([1 / 3, 1 / 3, 1 / 3, 0.0]).expand_as(books[..., 0, :])
    torch.testing.assert_close(books[..., 0, :], third, atol=0.03, rtol=0)
    torch.testing.assert_close(books[..., opened, :], torch.tensor([0.0, 1 / 3, 1 / 3, 1 / 3]).expand_as(third), atol=0.05, rtol=0)
    turnover = (td["action"][..., 1:] - td["portfolio_weights"][..., 1:]).abs().sum(-1)  # requested minus drifted
    assert turnover[..., 0].allclose(torch.full_like(turnover[..., 0], 2 / 3))
    assert turnover[..., opened].allclose(torch.full_like(turnover[..., opened], 1 / 3), atol=0.03, rtol=0)
    assert turnover[..., 1:opened].sum() == pytest.approx(0.0, abs=1e-6)
    assert turnover[..., opened + 1:].sum() == pytest.approx(0.0, abs=1e-6)


def test_ubah_under_a_gross_cap_keeps_the_held_lanes_when_a_lane_opens():
    """With max_gross below one, opening A2 must not make the env scale A0 and A1 down."""
    env = PortfolioTradingEnv(_staggered_bars(), PortfolioTradingEnvConfig(**SMALL, max_gross=0.5))
    opened = int(env.sampler.tradable_exec[:, 2].nonzero()[0])
    td = env.rollout(opened + 3, policy=UBAH(max_gross=0.5))
    books = td["next", "portfolio_weights"]
    torch.testing.assert_close(books[0], torch.tensor([2 / 3, 1 / 6, 1 / 6, 0.0]), atol=0.02, rtol=0)
    torch.testing.assert_close(books[opened, 1:3], books[opened - 1, 1:3], atol=0.01, rtol=0)  # held lanes drift only
    assert books[opened, 3] == pytest.approx(1 / 6, abs=0.02)
    assert td["action"][opened, 1:].sum() <= 0.5 + 1e-6  # the request respects the cap; the drifted book may not


def test_ubah_over_the_cap_is_sold_down_by_the_env_not_the_policy():
    """Two lanes rally past a cap below one: the policy echoes its book, the env trades it down to the cap."""
    bars = make_portfolio_bars()
    hours = bars.groupby("inst_id").cumcount()
    rally = (bars["inst_id"] != "A2") * 0.03 * hours  # A0 and A1 climb 3% an hour, A2 is flat
    for col in ("open", "high", "low", "close"):
        bars[col] = 100.0 * (1 + rally)
    env = PortfolioTradingEnv(bars, PortfolioTradingEnvConfig(**SMALL, max_gross=0.5))
    td = env.rollout(4, policy=UBAH(max_gross=0.5))
    torch.testing.assert_close(td["action"][1:], td["portfolio_weights"][1:])  # after the first fill, the policy only echoes
    excess = td["portfolio_weights"][1:, 1:].sum(-1) - 0.5
    assert (excess > 0.01).all()  # the book drifts above the cap every period
    torch.testing.assert_close(torch.tensor(env.history.turnovers[2:], dtype=torch.float64), excess.double(), atol=1e-3, rtol=0)  # the env sells exactly the excess
    assert (td["action"] >= 0).all()


def test_ubah_buys_equal_weights_once_then_holds():
    env = _env(False, transaction_fee=0.001)
    env.rollout(6, policy=UBAH())
    h = env.history.to_dict()
    assert h["turnovers"][1] == pytest.approx(1.0) and h["commissions"][1] > 0
    assert sum(h["turnovers"][2:]) == pytest.approx(0.0, abs=1e-6)  # float32 observation round trip
    torch.testing.assert_close(torch.tensor(h["weights"][1][1:]), torch.full((3,), 1 / 3), atol=0.05, rtol=0)
    # The float32 cash weight can round below zero after a listing; the echoed action must stay in spec.
    td = UBAH()(TensorDict({"portfolio_weights": torch.tensor([-2.2e-16, 0.5, 0.5, 0.0]), "tradable": torch.tensor([1.0, 1.0, 0.0])}))
    assert env.action_spec.is_in(td["action"]) and torch.equal(td["action"], torch.tensor([0.0, 0.5, 0.5, 0.0]))
    # A partially invested book (the fill under max_gross < 1) is held, not re-bought.
    td = UBAH()(TensorDict({"portfolio_weights": torch.tensor([0.5, 0.25, 0.25, 0.0]), "tradable": torch.tensor([1.0, 1.0, 0.0])}))
    torch.testing.assert_close(td["action"], torch.tensor([0.5, 0.25, 0.25, 0.0]))
    # Two lanes open at once with less cash than 2/N left: they share the cash, nothing is sold.
    td = UBAH()(TensorDict({"portfolio_weights": torch.tensor([0.4, 0.6, 0.0, 0.0]), "tradable": torch.tensor([1.0, 1.0, 1.0])}))
    torch.testing.assert_close(td["action"], torch.tensor([0.0, 0.6, 0.2, 0.2]))
    # Held lanes at or above the cap: the lane that opens stays unbought, nothing is sold or negative.
    for cap, book in ((0.5, [0.4, 0.3, 0.3, 0.0]), (1.0, [0.0, 0.5, 0.5, 0.0])):
        td = UBAH(max_gross=cap)(TensorDict({"portfolio_weights": torch.tensor(book), "tradable": torch.ones(3)}))
        torch.testing.assert_close(td["action"], torch.tensor(book))
    # A real book (sums to one) whose thirteen shares sum back to 2.4e-7 more than the cash in float32.
    w = torch.zeros(16); w[:3] = torch.tensor([0.843706429, 0.0434619002, 0.112831645])
    td = UBAH()(TensorDict({"portfolio_weights": w, "tradable": torch.ones(15)}))
    assert td["action"][0] == 0 and td["action"].sum() == pytest.approx(1.0, abs=1e-6)


def test_ucrp_rebalances_to_the_same_target_every_step():
    env = _env(False)
    td = env.rollout(4, policy=UCRP())
    torch.testing.assert_close(td["action"], torch.tensor([0.0, 1 / 3, 1 / 3, 1 / 3]).expand(4, -1))
    # Undoing drift moves weight between assets with zero net change: a signed turnover would read 0.
    assert all(t > 0 for t in env.history.turnovers[2:])


def _olmar_td(window_closes, tradable, weights):
    """One observation with a 6-bar window: `window_closes` is (N, 6) close over the latest close.
    High and low differ from close, and a decoy second market-data key holds a rolled window."""
    closes = torch.tensor(window_closes)
    market = torch.stack([closes, closes * 1.02, closes * 0.98], -1)
    return TensorDict({
        "market_data_1Hour_6": market, "market_data_1Day_6": market.roll(1, -2),
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
    # B is closed: its 0.8 history must not shift the mean (leaked: [0.877, 0.123, 0]); a mask
    # collapsed to one value per book would gate A too ([1, 0, 0]).
    pytest.param([A_FELL, B_ROSE], [1.0, 0.0], [1.0, 0.0, 0.0], 1.1, [0.7, 0.3, 0.0], id="closed-asset-does-not-shift-the-mean"),
    pytest.param([FLAT, FLAT], [1.0, 1.0], [0.0, -0.5, -0.5], 1.1, [1 / 3, 1 / 3, 1 / 3], id="all-short-book-starts-from-nothing"),
    pytest.param([[0.0] * 6, FLAT], [1.0, 1.0], [0.2, 0.3, 0.5], 1.1, [0.2, 0.3, 0.5], id="tradable-without-bars-is-no-signal"),
    pytest.param([FLAT, FLAT], [1.0, 1.0], [0.2, 0.3, 0.5], 1.1, [0.2, 0.3, 0.5], id="flat-window-keeps-the-book"),
    pytest.param([A_FELL, B_ROSE], [1.0, 1.0], [0.0, 1.0, 0.0], 1.1, [0.0, 1.0, 0.0], id="prediction-above-epsilon-no-move"),
    pytest.param([[0.0, 0.0, 0.0, 0.0, 1.2, 1.0]], [1.0], [1.0, 0.0], 1.05, [0.5, 0.5], id="mid-window-listing-counts-its-own-bars"),
    pytest.param([FLAT, FLAT], [1.0, 1.0], [0.2, 0.3, -0.5], 1.1, [0.4, 0.6, 0.0], id="short-book-clipped-and-renormalised"),
    # B is closed: its bars are no signal, so b.x_hat = 1.125 sits above epsilon and nothing moves.
    pytest.param([A_FELL, B_ROSE], [1.0, 0.0], [0.0, 0.5, 0.5], 1.1, [0.0, 0.5, 0.5], id="closed-asset-bars-are-no-signal"),
    # A 0.03% signal still moves all in: the denominator floor is a division guard, not damping.
    pytest.param([[1.0004] * 5 + [1.0]], [1.0], [1.0, 0.0], 1.1, [0.0, 1.0], id="tiny-signal-still-moves-all-in"),
])
def test_olmar_edge_cases(windows, tradable, weights, epsilon, expected):
    td = _olmar_td(windows, tradable, weights)
    OLMAR(window=4, epsilon=epsilon)(td)
    torch.testing.assert_close(td["action"], torch.tensor(expected))


def test_olmar_lanes_are_independent():
    """On the vectorized env every reduction is per lane: a batch of two different windows
    gives each lane the action it gets alone."""
    lanes = [
        _olmar_td([A_FELL, B_ROSE], [1.0, 1.0], [1 / 3, 1 / 3, 1 / 3]),
        _olmar_td([A_FELL, FLAT], [1.0, 1.0], [1 / 3, 1 / 3, 1 / 3]),
    ]
    policy = OLMAR(window=4, epsilon=1.1)
    batched = policy(torch.stack(lanes, 0))
    for i, lane in enumerate(lanes):
        torch.testing.assert_close(batched["action"][i], policy(lane)["action"])
