"""Exploration script for FuturesOneStepEnv with enhanced metrics.

This script is designed to investigate whether the GRPO algorithm is learning
by tracking detailed metrics about:
- Action distribution (long/short/hold breakdown)
- Reward statistics (mean, std, percentiles)
- Win/loss rates
- Policy entropy and behavior
- Gradient health
- Learning progress over time
"""
from __future__ import annotations

import warnings
from collections import deque

import hydra
from torchrl._utils import compile_with_warmup
import datasets
from torchtrade.losses import GRPOLoss
import numpy as np


def compute_action_statistics(actions, num_sl_tp_combinations):
    """Compute action distribution statistics.

    Action space:
    - 0: Hold
    - 1 to N: Long with SL/TP combinations
    - N+1 to 2N: Short with SL/TP combinations
    """
    actions_np = actions.cpu().numpy().flatten()
    total = len(actions_np)

    hold_count = (actions_np == 0).sum()
    long_count = ((actions_np >= 1) & (actions_np <= num_sl_tp_combinations)).sum()
    short_count = (actions_np > num_sl_tp_combinations).sum()

    return {
        "hold_pct": hold_count / total * 100,
        "long_pct": long_count / total * 100,
        "short_pct": short_count / total * 100,
        "hold_count": int(hold_count),
        "long_count": int(long_count),
        "short_count": int(short_count),
    }


def compute_reward_statistics(rewards):
    """Compute detailed reward statistics."""
    rewards_np = rewards.cpu().numpy().flatten()

    # Filter out zeros (hold actions)
    non_zero_rewards = rewards_np[rewards_np != 0]

    stats = {
        "mean": float(rewards_np.mean()),
        "std": float(rewards_np.std()),
        "min": float(rewards_np.min()),
        "max": float(rewards_np.max()),
        "median": float(np.median(rewards_np)),
    }

    if len(non_zero_rewards) > 0:
        stats["non_zero_mean"] = float(non_zero_rewards.mean())
        stats["win_rate"] = float((non_zero_rewards > 0).sum() / len(non_zero_rewards) * 100)
        stats["loss_rate"] = float((non_zero_rewards < 0).sum() / len(non_zero_rewards) * 100)
        stats["num_trades"] = int(len(non_zero_rewards))

        # Percentiles
        stats["p25"] = float(np.percentile(non_zero_rewards, 25))
        stats["p75"] = float(np.percentile(non_zero_rewards, 75))
    else:
        stats["non_zero_mean"] = 0.0
        stats["win_rate"] = 0.0
        stats["loss_rate"] = 0.0
        stats["num_trades"] = 0
        stats["p25"] = 0.0
        stats["p75"] = 0.0

    return stats


def compute_gradient_stats(model):
    """Compute gradient statistics to monitor training health."""
    total_norm = 0.0
    param_count = 0
    grad_min = float('inf')
    grad_max = float('-inf')

    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2).item()
            total_norm += param_norm ** 2
            param_count += 1
            grad_min = min(grad_min, p.grad.data.min().item())
            grad_max = max(grad_max, p.grad.data.max().item())

    total_norm = total_norm ** 0.5

    return {
        "grad_norm": total_norm,
        "grad_min": grad_min if param_count > 0 else 0.0,
        "grad_max": grad_max if param_count > 0 else 0.0,
    }


class MovingAverage:
    """Track moving averages for trend analysis."""

    def __init__(self, window_size=100):
        self.window_size = window_size
        self.values = deque(maxlen=window_size)

    def update(self, value):
        self.values.append(value)

    def mean(self):
        if len(self.values) == 0:
            return 0.0
        return sum(self.values) / len(self.values)

    def std(self):
        if len(self.values) < 2:
            return 0.0
        mean = self.mean()
        return (sum((x - mean) ** 2 for x in self.values) / len(self.values)) ** 0.5


@hydra.main(config_path="", config_name="config", version_base="1.1")
def main(cfg: DictConfig):  # noqa: F821

    import torch
    import torch.optim
    import tqdm
    import wandb
    from tensordict import TensorDict
    from tensordict.nn import CudaGraphModule

    from torchrl._utils import timeit
    from torchrl.envs import ExplorationType, set_exploration_type
    from torchrl.record.loggers import generate_exp_name, get_logger
    from utils import make_environment, make_grpo_policy, make_collector, log_metrics

    torch.set_float32_matmul_precision("high")

    device = cfg.optim.device
    if device in ("", None):
        if torch.cuda.is_available():
            device = "cuda:0"
        else:
            device = "cpu"
    device = torch.device(device)
    print("=" * 60)
    print("FUTURES ONE-STEP ENVIRONMENT EXPLORATION")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Leverage: {cfg.env.leverage}x")
    print(f"SL levels: {cfg.env.stoploss_levels}")
    print(f"TP levels: {cfg.env.takeprofit_levels}")

    # Calculate number of SL/TP combinations
    num_sl = len(cfg.env.stoploss_levels)
    num_tp = len(cfg.env.takeprofit_levels)
    num_sl_tp_combinations = num_sl * num_tp
    num_actions = 1 + 2 * num_sl_tp_combinations  # hold + long + short
    print(f"Action space: {num_actions} actions (1 hold + {num_sl_tp_combinations} long + {num_sl_tp_combinations} short)")
    print("=" * 60)

    # Load data from HuggingFace
    df = datasets.load_dataset(cfg.env.data_path)
    df = df["train"].to_pandas()
    test_df = df[0 : (1440 * 30)]  # 21 days for test
    train_df = df[(1440 * 30) :]
    print(f"Training data: {len(train_df)} rows")
    print(f"Test data: {len(test_df)} rows")

    max_train_traj_length = cfg.collector.frames_per_batch // cfg.env.train_envs
    max_eval_traj_length = len(test_df)
    train_env, eval_env = make_environment(
        train_df,
        test_df,
        cfg,
        train_num_envs=cfg.env.train_envs,
        eval_num_envs=cfg.env.eval_envs,
        max_train_traj_length=max_train_traj_length,
        max_eval_traj_length=max_eval_traj_length,
    )
    eval_env.to(device)

    total_frames = cfg.collector.total_frames
    frames_per_batch = cfg.collector.frames_per_batch
    test_interval = cfg.logger.test_interval

    compile_mode = None
    if cfg.compile.compile:
        compile_mode = cfg.compile.compile_mode
        if compile_mode in ("", None):
            if cfg.compile.cudagraphs:
                compile_mode = "default"
            else:
                compile_mode = "reduce-overhead"

    # Create policy
    actor = make_grpo_policy(
        eval_env,
        device=device,
        cfg=cfg,
    )

    # Create collector
    collector = make_collector(
        cfg,
        train_env,
        actor,
        compile_mode,
    )

    # Create loss module
    loss_module = GRPOLoss(
        actor_network=actor,
        entropy_coeff=cfg.loss.entropy_coef,
    )

    # Create optimizer
    optim = torch.optim.Adam(
        loss_module.parameters(),
        lr=cfg.optim.lr,
    )

    # Create logger
    logger = None
    if cfg.logger.backend:
        exp_name = generate_exp_name("TorchTrade-FuturesOneStep-Explore", cfg.logger.exp_name)
        logger = get_logger(
            cfg.logger.backend,
            logger_name="grpo_futures_explore",
            experiment_name=exp_name,
            wandb_kwargs={
                "mode": cfg.logger.mode,
                "config": dict(cfg),
                "project": cfg.logger.project_name,
                "group": cfg.logger.group_name,
            },
        )

    # Initialize moving averages for trend tracking
    reward_ma = MovingAverage(window_size=50)
    loss_ma = MovingAverage(window_size=50)
    entropy_ma = MovingAverage(window_size=50)
    win_rate_ma = MovingAverage(window_size=50)

    # Main loop
    collected_frames = 0
    best_eval_reward = float('-inf')
    pbar = tqdm.tqdm(total=total_frames)

    def update(batch):
        optim.zero_grad(set_to_none=True)
        batch = batch.to(device, non_blocking=True)

        # Forward pass GRPO loss
        loss_td = loss_module(batch)
        loss = loss_td["loss_objective"] + loss_td["loss_entropy"]

        # Backward pass
        loss.backward()

        # Compute gradient stats before stepping
        grad_stats = compute_gradient_stats(loss_module)

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(loss_module.parameters(), max_norm=1.0)

        # Update the networks
        optim.step()

        loss_td = loss_td.detach()
        loss_td.set("grad_norm", torch.tensor(grad_stats["grad_norm"]))
        loss_td.set("grad_min", torch.tensor(grad_stats["grad_min"]))
        loss_td.set("grad_max", torch.tensor(grad_stats["grad_max"]))

        return loss_td

    if cfg.compile.compile:
        update = compile_with_warmup(update, mode=compile_mode, warmup=1)

    if cfg.compile.cudagraphs:
        warnings.warn(
            "CudaGraphModule is experimental and may lead to silently wrong results. Use with caution.",
            category=UserWarning,
        )
        update = CudaGraphModule(update, in_keys=[], out_keys=[], warmup=5)

    collector_iter = iter(collector)
    total_iter = len(collector)

    print("\nStarting training loop...")
    print("-" * 60)

    for i in range(total_iter):
        timeit.printevery(1000, total_iter, erase=True)

        with timeit("collecting"):
            data = next(collector_iter).to(device)

        metrics_to_log = {}
        frames_in_batch = data.numel()
        collected_frames += frames_in_batch
        pbar.update(frames_in_batch)

        # ================================================================
        # Action Distribution Analysis
        # ================================================================
        actions = data["action"]
        action_stats = compute_action_statistics(actions, num_sl_tp_combinations)

        metrics_to_log["actions/hold_pct"] = action_stats["hold_pct"]
        metrics_to_log["actions/long_pct"] = action_stats["long_pct"]
        metrics_to_log["actions/short_pct"] = action_stats["short_pct"]
        metrics_to_log["actions/action_std"] = data["action"].float().std().item()
        metrics_to_log["actions/action_mean"] = data["action"].float().mean().item()

        # ================================================================
        # Reward Analysis
        # ================================================================
        rewards = data["next", "reward"]
        reward_stats = compute_reward_statistics(rewards)

        metrics_to_log["reward/mean"] = reward_stats["mean"]
        metrics_to_log["reward/std"] = reward_stats["std"]
        metrics_to_log["reward/min"] = reward_stats["min"]
        metrics_to_log["reward/max"] = reward_stats["max"]
        metrics_to_log["reward/median"] = reward_stats["median"]
        metrics_to_log["reward/non_zero_mean"] = reward_stats["non_zero_mean"]
        metrics_to_log["reward/win_rate"] = reward_stats["win_rate"]
        metrics_to_log["reward/loss_rate"] = reward_stats["loss_rate"]
        metrics_to_log["reward/num_trades"] = reward_stats["num_trades"]
        metrics_to_log["reward/p25"] = reward_stats["p25"]
        metrics_to_log["reward/p75"] = reward_stats["p75"]

        # Update moving averages
        reward_ma.update(reward_stats["mean"])
        if reward_stats["num_trades"] > 0:
            win_rate_ma.update(reward_stats["win_rate"])

        metrics_to_log["reward/ma_mean"] = reward_ma.mean()
        metrics_to_log["reward/ma_win_rate"] = win_rate_ma.mean()

        # ================================================================
        # Training Update
        # ================================================================
        with timeit("training"):
            with timeit("update"):
                torch.compiler.cudagraph_mark_step_begin()
                loss = update(data)
            loss = loss.clone()

        # Loss metrics
        metrics_to_log["loss/objective"] = loss["loss_objective"].item()
        metrics_to_log["loss/entropy"] = loss["loss_entropy"].item()
        metrics_to_log["loss/total"] = loss["loss_objective"].item() + loss["loss_entropy"].item()

        if "entropy" in loss.keys():
            metrics_to_log["policy/entropy"] = loss["entropy"].item()
            entropy_ma.update(loss["entropy"].item())
            metrics_to_log["policy/entropy_ma"] = entropy_ma.mean()

        if "kl_approx" in loss.keys():
            metrics_to_log["policy/kl_approx"] = loss["kl_approx"].item()

        if "advantage" in loss.keys():
            metrics_to_log["policy/advantage"] = loss["advantage"].item()

        # Gradient metrics
        if "grad_norm" in loss.keys():
            metrics_to_log["grad/norm"] = loss["grad_norm"].item()
            metrics_to_log["grad/min"] = loss["grad_min"].item()
            metrics_to_log["grad/max"] = loss["grad_max"].item()

        # Update loss moving average
        loss_ma.update(loss["loss_objective"].item())
        metrics_to_log["loss/ma_objective"] = loss_ma.mean()

        # ================================================================
        # Evaluation
        # ================================================================
        with torch.no_grad(), set_exploration_type(
            ExplorationType.DETERMINISTIC
        ), timeit("eval"):
            if ((i - 1) * frames_in_batch) // test_interval < (
                i * frames_in_batch
            ) // test_interval:
                actor.eval()
                eval_rollout = eval_env.rollout(
                    max_eval_traj_length,
                    actor,
                    auto_cast_to_device=True,
                    break_when_any_done=True,
                )
                eval_rollout.squeeze()
                eval_reward = eval_rollout["next", "reward"].sum(-2).mean().item()
                metrics_to_log["eval/reward"] = eval_reward

                # Track best evaluation
                if eval_reward > best_eval_reward:
                    best_eval_reward = eval_reward
                    metrics_to_log["eval/best_reward"] = best_eval_reward
                    # Save best model
                    torch.save(actor.state_dict(), "best_policy.pth")
                    print(f"\n🏆 New best eval reward: {eval_reward:.4f}")

                # Eval action distribution
                eval_actions = eval_rollout["action"]
                eval_action_stats = compute_action_statistics(eval_actions, num_sl_tp_combinations)
                metrics_to_log["eval/hold_pct"] = eval_action_stats["hold_pct"]
                metrics_to_log["eval/long_pct"] = eval_action_stats["long_pct"]
                metrics_to_log["eval/short_pct"] = eval_action_stats["short_pct"]

                # Eval reward stats
                eval_rewards = eval_rollout["next", "reward"]
                eval_reward_stats = compute_reward_statistics(eval_rewards)
                metrics_to_log["eval/win_rate"] = eval_reward_stats["win_rate"]
                metrics_to_log["eval/num_trades"] = eval_reward_stats["num_trades"]

                # Note: render_history is not available for FuturesOneStepEnv
                # (only SeqFuturesEnv has it, but we use FuturesOneStepEnv for consistency)
                eval_env.reset()

                actor.train()

                # Print progress summary
                print(f"\n[Iter {i}] Frames: {collected_frames:,}")
                print(f"  Train - Reward: {reward_stats['mean']:.4f}, Win: {reward_stats['win_rate']:.1f}%, Trades: {reward_stats['num_trades']}")
                print(f"  Eval  - Reward: {eval_reward:.4f}, Win: {eval_reward_stats['win_rate']:.1f}%, Trades: {eval_reward_stats['num_trades']}")
                print(f"  Actions - Hold: {action_stats['hold_pct']:.1f}%, Long: {action_stats['long_pct']:.1f}%, Short: {action_stats['short_pct']:.1f}%")
                if "entropy" in loss.keys():
                    print(f"  Policy - Entropy: {loss['entropy'].item():.4f}, Loss: {loss['loss_objective'].item():.4f}")

        # ================================================================
        # Logging
        # ================================================================
        if logger is not None:
            time_dict = timeit.todict(prefix="time")
            metrics_to_log.update(timeit.todict(prefix="time"))
            metrics_to_log["time/speed"] = pbar.format_dict["rate"]
            if time_dict["time/collecting"] > 0:
                metrics_to_log["time/SPS-collecting"] = (
                    frames_in_batch / time_dict["time/collecting"]
                )
            metrics_to_log["time/SPS-total"] = frames_in_batch / max(sum(time_dict.values()), 1e-6)
            metrics_to_log["progress/frames"] = collected_frames
            metrics_to_log["progress/iterations"] = i
            log_metrics(logger, metrics_to_log, collected_frames)

        collector.update_policy_weights_()

    # ================================================================
    # Final Summary
    # ================================================================
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Total frames collected: {collected_frames:,}")
    print(f"Best eval reward: {best_eval_reward:.4f}")
    print(f"Final reward MA: {reward_ma.mean():.4f}")
    print(f"Final win rate MA: {win_rate_ma.mean():.1f}%")
    print(f"Final entropy MA: {entropy_ma.mean():.4f}")
    print("=" * 60)

    # Save final model
    torch.save(actor.state_dict(), "final_policy.pth")
    print("Saved final_policy.pth and best_policy.pth")

    collector.shutdown()
    if not eval_env.is_closed:
        eval_env.close()
    if not train_env.is_closed:
        train_env.close()


if __name__ == "__main__":
    main()
