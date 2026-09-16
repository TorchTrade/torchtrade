"""Evaluate the three portfolio baselines (UBAH, UCRP, OLMAR) on the OKX multi-asset dataset.

Rolls each baseline through `PortfolioTradingEnv` over the whole timeline at a taker fee,
prints a metrics table and saves the portfolio value curves. Compare an RL policy against
these numbers by passing it as `policy=` to the same rollout.

Usage:
    python examples/offline/portfolio_baselines.py [--fee 0.0005] [--plot portfolio_baselines.png]
"""

import argparse

import matplotlib
import matplotlib.pyplot as plt

from torchtrade.actor import OLMAR, UBAH, UCRP
from torchtrade.envs.offline import PortfolioTradingEnv, PortfolioTradingEnvConfig
from torchtrade.envs.offline.infrastructure.utils import load_portfolio_dataset
from torchtrade.metrics import portfolio_metrics

matplotlib.use("Agg")
COLUMNS = ["final_value", "sharpe_ratio", "max_drawdown", "turnover", "commission", "funding"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fee", type=float, default=0.0005, help="taker fee per unit notional")
    parser.add_argument("--plot", default="portfolio_baselines.png")
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
    curves = {}
    for name, policy in [("UBAH", UBAH()), ("UCRP", UCRP()), ("OLMAR", OLMAR(window=5, epsilon=10.0))]:
        env.rollout(env.sampler.num_exec, policy=policy)
        m = portfolio_metrics(env.history, periods_per_year)
        curves[name] = (env.history.timestamps, env.history.portfolio_values)
        print(f"| {name} | " + " | ".join(f"{m[c]:.3f}" for c in COLUMNS) + " |")

    fig, ax = plt.subplots(figsize=(9, 4))
    for name, (ts, pv) in curves.items():
        ax.plot(ts, [v / pv[0] for v in pv], label=name)
    ax.set_ylabel("portfolio value / initial")
    ax.set_title(f"OKX multi-asset 1h, 4h decisions, fee {args.fee:.2%}")
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(args.plot, dpi=120)
    print(f"saved {args.plot}")


if __name__ == "__main__":
    main()
