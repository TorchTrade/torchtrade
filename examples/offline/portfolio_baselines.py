"""Evaluate the three portfolio baselines (UBAH, UCRP, OLMAR) on the OKX multi-asset dataset.

Rolls each baseline through `PortfolioTradingEnv` over the whole timeline at 1h, 4h and 1d
decisions at a taker fee, prints one metrics table and saves the equity curves as three
stacked panels (`portfolio_baselines.png`, the figure in docs/environments/portfolio.md).
Compare an RL policy against these numbers by passing it as `policy=` to the same rollout.

Usage:
    python examples/offline/portfolio_baselines.py [--fee 0.0005]
"""

import argparse

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
import seaborn as sns

from torchtrade.actor import OLMAR, UBAH, UCRP
from torchtrade.envs.offline import PortfolioTradingEnv, PortfolioTradingEnvConfig
from torchtrade.envs.offline.infrastructure.utils import load_portfolio_dataset
from torchtrade.metrics import portfolio_metrics

COLUMNS = ["final_value", "sharpe_ratio", "max_drawdown", "turnover", "commission", "funding"]
FREQUENCIES = [("1Hour", 24 * 365), ("4Hour", 6 * 365), ("1Day", 365)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fee", type=float, default=0.0005, help="taker fee per unit notional (OKX regular tier)")
    args = parser.parse_args()
    bars, _, funding = load_portfolio_dataset()

    sns.set()
    fig, axes = plt.subplots(len(FREQUENCIES), 1, figsize=(15, 5.5 * len(FREQUENCIES)), sharex=True)
    print(f"| decisions | baseline | {' | '.join(COLUMNS)} |")
    print(f"|---|---|{'---|' * len(COLUMNS)}")
    for ax, (execute_on, periods_per_year) in zip(axes, FREQUENCIES):
        config = PortfolioTradingEnvConfig(
            time_frames=["1Hour"], window_sizes=[50], execute_on=execute_on,
            transaction_fee=args.fee, random_start=False,
        )
        env = PortfolioTradingEnv(bars, config, funding=funding)
        curves = {}
        for policy in (UBAH(), UCRP(), OLMAR()):
            name = type(policy).__name__
            env.rollout(env.sampler.num_exec, policy=policy)
            m = portfolio_metrics(env.history, periods_per_year)
            row = " | ".join(f"{m[c]:.3f}" for c in COLUMNS)
            print(f"| {execute_on} | {name} | {row} |")
            pv = env.history.portfolio_values
            curves[f"{name} ({(pv[-1] / pv[0] - 1) * 100:+.1f}%)"] = [(v / pv[0] - 1) * 100 for v in pv]
        table = pd.DataFrame(curves, index=pd.to_datetime(env.history.timestamps)).rename_axis("Time")
        long = table.reset_index().melt(id_vars="Time", var_name="Series", value_name="Cumulative return")
        sns.lineplot(x="Time", y="Cumulative return", hue="Series", data=long, drawstyle="steps-post", linewidth=2.5, ax=ax)
        ax.set_title(f"{execute_on} decisions ({len(table) - 1} steps)", fontsize=15)
        ax.set_ylabel("Cumulative return (% of initial value)", fontsize=15)
        ax.set_xlabel("Time (UTC)" if execute_on == FREQUENCIES[-1][0] else "", fontsize=15)
        ax.tick_params(labelsize=12)
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:+.0f}%"))
        ax.axhline(0, color="black", linewidth=0.8)
        ax.legend(loc="upper left" if execute_on == "1Day" else "lower left", fontsize=13)

    fig.suptitle(f"OKX multi-asset 1h bars, 40 perpetual swaps, fee {args.fee:.2%}, funding charged", fontsize=17)
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    fig.savefig("portfolio_baselines.png", dpi=150)
    print("saved portfolio_baselines.png")


if __name__ == "__main__":
    main()
