"""Plot the LLM-actor vLLM benchmark results (DGX Spark) for the findings doc / X post.

Reads a combined CSV (model,engine,N,wall_s,decisions_per_s,tokens_per_s,p50_ms,p95_ms,p99_ms)
and produces publication-quality PNGs. Engines get fixed, colorblind-safe (Okabe-Ito)
colors so identity is never color-alone-ambiguous; the batching "hero" is emphasized.

    python benchmarks/plot_bench.py benchmarks/data/spark_combined.csv --out benchmarks/plots
"""
from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Okabe-Ito, CVD-safe. Fixed order by engine identity (not by rank).
ENGINE_STYLE = {
    "ours-current":  ("#999999", "o", "TorchTrade sequential (before PR)"),
    "ours-batched":  ("#0072B2", "s", "TorchTrade batched (this PR)"),
    "torchrl-sync":  ("#E69F00", "^", "TorchRL-native vLLMWrapper"),
    "torchrl-async": ("#009E73", "D", "TorchRL-native AsyncVLLM"),
}
MODEL_ORDER = ["Qwen/Qwen2.5-0.5B-Instruct", "Qwen/Qwen2.5-3B-Instruct", "Qwen/Qwen2.5-7B-Instruct"]
MODEL_LABEL = {m: m.split("/")[-1].replace("-Instruct", "") for m in MODEL_ORDER}


def load(path):
    # rows[model][engine][N] = dict of metrics
    rows = defaultdict(lambda: defaultdict(dict))
    with open(path) as f:
        for r in csv.DictReader(f):
            try:
                rows[r["model"]][r["engine"]][int(r["N"])] = {
                    k: float(r[k]) for k in
                    ("wall_s", "decisions_per_s", "tokens_per_s", "p50_ms", "p95_ms", "p99_ms")
                }
            except (KeyError, ValueError):
                continue
    return rows


def _style_ax(ax):
    ax.grid(True, which="both", color="#dddddd", linewidth=0.6, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#888888")
    ax.tick_params(colors="#444444", labelsize=9)


def plot_scaling(rows, out, metric="tokens_per_s", ylabel="tokens / sec"):
    models = [m for m in MODEL_ORDER if m in rows]
    fig, axes = plt.subplots(1, len(models), figsize=(4.6 * len(models), 4.2), squeeze=False)
    axes = axes[0]
    for ax, model in zip(axes, models):
        for engine, (color, marker, _lab) in ENGINE_STYLE.items():
            if engine not in rows[model]:
                continue
            pts = sorted(rows[model][engine].items())
            xs = [n for n, _ in pts]
            ys = [d[metric] for _, d in pts]
            ax.plot(xs, ys, color=color, marker=marker, markersize=6, linewidth=2,
                    zorder=3, clip_on=False)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlabel("batch size N (parallel decisions)", fontsize=9)
        ax.set_title(MODEL_LABEL[model], fontsize=11, fontweight="bold", color="#222222")
        _style_ax(ax)
    axes[0].set_ylabel(ylabel, fontsize=10)
    handles = [plt.Line2D([], [], color=c, marker=m, markersize=6, linewidth=2, label=lab)
               for _e, (c, m, lab) in ENGINE_STYLE.items()
               if any(_e in rows[mm] for mm in models)]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False,
               fontsize=9, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(f"Batched LLM inference throughput ({ylabel}) — DGX Spark GB10",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    p = os.path.join(out, f"fig_scaling_{metric}.png")
    fig.savefig(p, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return p


def plot_batching_win(rows, out, at_n=128, metric="tokens_per_s", ylabel="tokens / sec"):
    """Hero: TorchTrade sequential vs batched at a large batch, per model (log y)."""
    models = [m for m in MODEL_ORDER if m in rows]
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    width = 0.38
    xs = range(len(models))
    for k, engine in enumerate(("ours-current", "ours-batched")):
        color, _m, lab = ENGINE_STYLE[engine]
        vals = []
        for m in models:
            cell = rows[m].get(engine, {})
            # use the requested N, else the largest available N for that engine
            if at_n in cell:
                vals.append(cell[at_n][metric])
            elif cell:
                vals.append(cell[max(cell)][metric])
            else:
                vals.append(0)
        bars = ax.bar([x + (k - 0.5) * width for x in xs], vals, width,
                      color=color, label=lab, zorder=3)
        for b, v in zip(bars, vals):
            if v > 0:
                ax.text(b.get_x() + b.get_width() / 2, v * 1.05, f"{v:.0f}",
                        ha="center", va="bottom", fontsize=9, color="#222222", fontweight="bold")
    ax.set_yscale("log")
    ax.set_xticks(list(xs))
    ax.set_xticklabels([MODEL_LABEL[m] for m in models], fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(f"Batching {metric.replace('_', ' ')} at N={at_n} — TorchTrade actor, DGX Spark",
                 fontsize=12, fontweight="bold", color="#222222")
    ax.set_ylim(top=ax.get_ylim()[1] * 3)  # headroom so labels/legend don't collide
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    _style_ax(ax)
    fig.tight_layout()
    p = os.path.join(out, f"fig_batching_win_{metric}.png")
    fig.savefig(p, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--out", default="benchmarks/plots")
    ap.add_argument("--at-n", type=int, default=128)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    rows = load(args.csv)
    made = [
        plot_scaling(rows, args.out, "tokens_per_s", "tokens / sec"),
        plot_scaling(rows, args.out, "decisions_per_s", "decisions / sec"),
        plot_batching_win(rows, args.out, args.at_n, "tokens_per_s", "tokens / sec"),
    ]
    for p in made:
        print("wrote", p)


if __name__ == "__main__":
    main()
