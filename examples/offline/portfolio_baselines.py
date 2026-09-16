"""Evaluate the three portfolio baselines (UBAH, UCRP, OLMAR) on the OKX multi-asset dataset.

Rolls each baseline through `PortfolioTradingEnv` over the whole timeline at a taker fee,
prints a metrics table and saves the portfolio value curves. Compare an RL policy against
these numbers by passing it as `policy=` to the same rollout.

Usage:
    python examples/offline/portfolio_baselines.py [--fee 0.0005]
"""

import argparse

import matplotlib.pyplot as plt

from torchtrade.actor import OLMAR, UBAH, UCRP
from torchtrade.envs.offline import PortfolioTradingEnv, PortfolioTradingEnvConfig
from torchtrade.envs.offline.infrastructure.utils import load_portfolio_dataset
from torchtrade.metrics import portfolio_metrics

COLUMNS = ["final_value", "sharpe_ratio", "max_drawdown", "turnover", "commission", "funding"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fee", type=float, default=0.0005, help="taker fee per unit notional")
    args = parser.parse_args()

    bars, _, funding = load_portfolio_dataset()
    config = PortfolioTradingEnvConfig(
        time_frames=["1Hour"], window_sizes=[50], execute_on="4Hour",
        transaction_fee=args.fee, random_start=False,
    )
    env = PortfolioTradingEnv(bars, config, funding=funding)
    periods_per_year = 6 * 365  # 4-hour decision bars

    print(f"| baseline | {' | '.join(COLUMNS)} |")
    print(f"|---|{'---|' * len(COLUMNS)}")
    fig, ax = plt.subplots(figsize=(9, 4))
    for policy in (UBAH(), UCRP(), OLMAR()):
        name = type(policy).__name__
        env.rollout(env.sampler.num_exec, policy=policy)
        m = portfolio_metrics(env.history, periods_per_year)
        pv = env.history.portfolio_values
        ax.plot(env.history.timestamps, [v / pv[0] for v in pv], label=name)
        row = " | ".join(f"{m[c]:.3f}" for c in COLUMNS)
        print(f"| {name} | {row} |")

    ax.set_ylabel("portfolio value / initial")
    ax.set_title(f"OKX multi-asset 1h, 4h decisions, fee {args.fee:.2%}")
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig("portfolio_baselines.png", dpi=120)
    print("saved portfolio_baselines.png")


if __name__ == "__main__":
    main()
