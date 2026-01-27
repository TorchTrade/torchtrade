"""DQN training example for SequentialTradingEnv.

This script demonstrates DQN training on the SequentialTradingEnv environment
for trading with optional leverage support. DQN is an off-policy algorithm
that uses experience replay and target networks for stable learning.

Usage:
    # Default (futures with 6x leverage)
    python train.py

    # Spot trading
    python train.py env=sequential

    # Custom leverage
    python train.py env.leverage=10
"""
from __future__ import annotations

import warnings

import hydra
import pandas as pd
from torchrl._utils import compile_with_warmup
import datasets


@hydra.main(config_path=".", config_name="config", version_base="1.1")
def main(cfg: DictConfig):  # noqa: F821

    import torch.optim
    import tqdm
    import wandb

    from torchrl._utils import timeit
    from torchrl.envs import ExplorationType, set_exploration_type
    from torchrl.record.loggers import generate_exp_name, get_logger
    from utils import (
        make_environment,
        make_tdqn_agent,
        make_collector,
        make_replay_buffer,
        make_loss_module,
        make_optimizer,
        log_metrics,
    )

    torch.set_float32_matmul_precision("high")

    device = cfg.optim.device
    if device in ("", None):
        if torch.cuda.is_available():
            device = "cuda:0"
        else:
            device = "cpu"
    device = torch.device(device)
    print("USING DEVICE: ", device)

    # Load data from HuggingFace
    df = datasets.load_dataset(cfg.env.data_path)
    df = df["train"].to_pandas()

    # Convert timestamp column to datetime for proper filtering
    df['0'] = pd.to_datetime(df['0'])
    test_split_date = pd.to_datetime(cfg.env.test_split_start)

    train_df = df[df['0'] < test_split_date]
    test_df  = df[df['0'] >= test_split_date]

    max_train_traj_length = 1000
    max_eval_traj_length = len(test_df)

    print("="*80)
    print("DATA SPLIT INFO:")
    print(f"Total rows: {len(df)}")
    print(f"Train rows (1min): {len(train_df)}")
    print(f"Test rows (1min): {len(test_df)}")
    print(f"Train date range: {train_df['0'].min()} to {train_df['0'].max()}")
    print(f"Test date range: {test_df['0'].min()} to {test_df['0'].max()}")
    print("="*80)
    train_env, eval_env, coverage_tracker = make_environment(
        train_df,
        test_df,
        cfg,
        train_num_envs=cfg.env.train_envs,
        eval_num_envs=cfg.env.eval_envs,
        max_train_traj_length=max_train_traj_length,
        max_eval_traj_length=max_eval_traj_length,
    )

    total_frames = cfg.collector.total_frames
    test_interval = cfg.logger.test_interval

    compile_mode = None
    if cfg.compile.compile:
        compile_mode = cfg.compile.compile_mode
        if compile_mode in ("", None):
            if cfg.compile.cudagraphs:
                compile_mode = "default"
            else:
                compile_mode = "reduce-overhead"


    actor, exploration_policy = make_tdqn_agent(
        cfg,
        train_env,
        device=device,
    )

    # Create collector with coverage tracker as postproc
    collector = make_collector(
        cfg,
        train_env,
        exploration_policy,
        compile_mode,
        postproc=coverage_tracker,
    )

    # Create replay buffer
    replay_buffer = make_replay_buffer(cfg, device=device)

    # Create loss module and target updater
    loss_module, target_updater = make_loss_module(cfg, actor)

    # Create optimizer
    optim = make_optimizer(cfg, loss_module)

    # Create logger
    logger = None
    if cfg.logger.backend:
        exp_name = generate_exp_name("TorchTrade-DQN", cfg.logger.exp_name)
        logger = get_logger(
            cfg.logger.backend,
            logger_name="dqn_logging",
            experiment_name=exp_name,
            wandb_kwargs={
                "mode": cfg.logger.mode,
                "config": dict(cfg),
                "project": cfg.logger.project_name,
                "group": cfg.logger.group_name,
            },
        )

    # Main loop
    collected_frames = 0
    pbar = tqdm.tqdm(total=total_frames)

    def update(batch):
        """Single optimization step."""
        optim.zero_grad(set_to_none=True)

        batch = batch.to(device, non_blocking=True)

        # Forward pass DQN loss
        loss_td = loss_module(batch)
        loss_sum = loss_td["loss"]

        # Backward pass
        loss_sum.backward()
        torch.nn.utils.clip_grad_norm_(
            loss_module.parameters(), max_norm=cfg.optim.max_grad_norm
        )

        # Update the networks
        optim.step()
        return loss_td.detach()

    if cfg.compile.compile:
        update = compile_with_warmup(update, mode=compile_mode, warmup=1)

    if cfg.compile.cudagraphs:
        warnings.warn(
            "CudaGraphModule is experimental and may lead to silently wrong results. Use with caution.",
            category=UserWarning,
        )
        from tensordict.nn import CudaGraphModule
        update = CudaGraphModule(update, in_keys=[], out_keys=[], warmup=5)

    collector_iter = iter(collector)
    total_iter = len(collector)
    for i in range(total_iter):
        timeit.printevery(1000, total_iter, erase=True)

        with timeit("collecting"):
            data = next(collector_iter).to(device)

        metrics_to_log = {}
        frames_in_batch = data.numel()
        collected_frames += frames_in_batch
        pbar.update(frames_in_batch)

        # Get training rewards and episode lengths
        episode_rewards = data["next", "episode_reward"][data["next", "terminated"]]
        if len(episode_rewards) > 0:
            episode_length = data["next", "step_count"][data["next", "terminated"]]
            metrics_to_log.update(
                {
                    "train/reward": episode_rewards.mean().item(),
                    "train/episode_length": episode_length.sum().item()
                    / len(episode_length),
                }
            )

        # Store in replay buffer
        with timeit("rb - extend"):
            data_reshape = data.reshape(-1)
            replay_buffer.extend(data_reshape)

        # Training updates
        with timeit("training"):
            # Only start training after initial random frames
            if collected_frames >= cfg.collector.init_random_frames:
                num_updates = cfg.loss.num_updates
                losses = []
                for _ in range(num_updates):
                    with timeit("rb - sample"):
                        batch = replay_buffer.sample()
                    with timeit("update"):
                        torch.compiler.cudagraph_mark_step_begin()
                        loss_td = update(batch)
                    losses.append(loss_td["loss"].item())

                    # Update target network
                    target_updater.step()

                # Log training losses
                if losses:
                    metrics_to_log["train/loss"] = sum(losses) / len(losses)

        # Log epsilon for exploration
        exploration_policy[-1].step(collected_frames)
        current_epsilon = exploration_policy[-1].eps
        metrics_to_log["train/epsilon"] = current_epsilon

        # Evaluation
        with torch.no_grad(), set_exploration_type(
            ExplorationType.DETERMINISTIC
        ), timeit("eval"):
            if ((i - 1) * frames_in_batch) // test_interval < (
                i * frames_in_batch
            ) // test_interval:
                actor.eval()
                # Move actor to CPU temporarily for eval
                eval_rollout = eval_env.rollout(
                    max_eval_traj_length,
                    actor.to("cpu"),
                    auto_cast_to_device=False,
                    break_when_any_done=True,
                )
                # Move actor back to device for training
                actor.to(device)
                eval_rollout.squeeze()
                eval_reward = eval_rollout["next", "reward"].sum(-2).mean().item()
                metrics_to_log["eval/reward"] = eval_reward

                # Compute and log trading metrics
                try:
                    # ParallelEnv delegates get_metrics() to all workers and returns a list
                    # We take the first environment's metrics
                    env_metrics = eval_env.base_env.get_metrics()[0]
                    metrics_to_log.update({f"eval/{k}": v for k, v in env_metrics.items()})

                except (KeyError, AttributeError, ValueError, RuntimeError) as e:
                    import traceback
                    print(f"Warning: Could not compute metrics: {e}")
                    print(traceback.format_exc())

                # Render history
                fig = eval_env.base_env.render_history(return_fig=True)
                eval_env.reset()
                if fig is not None and logger is not None:
                    metrics_to_log["eval/history"] = wandb.Image(fig[0])
                torch.save(actor.state_dict(), f"dqn_policy_{i}.pth")
                actor.train()

        # Log dual coverage metrics (if available and enabled)
        if coverage_tracker is not None:
            coverage_stats = coverage_tracker.get_coverage_stats()
            if coverage_stats["enabled"]:
                # Reset coverage (episode start diversity)
                metrics_to_log["train/reset_coverage"] = coverage_stats["reset_coverage"]
                # State coverage (full trajectory coverage)
                metrics_to_log["train/state_coverage"] = coverage_stats["state_coverage"]

        if logger is not None:
            time_dict = timeit.todict(prefix="time")
            metrics_to_log.update(timeit.todict(prefix="time"))
            metrics_to_log["time/speed"] = pbar.format_dict["rate"]
            metrics_to_log["time/SPS-collecting"] = (
                frames_in_batch / time_dict["time/collecting"]
            )
            metrics_to_log["time/SPS-total"] = frames_in_batch / sum(time_dict.values())
            log_metrics(logger, metrics_to_log, collected_frames)

        collector.update_policy_weights_()

    collector.shutdown()
    if not eval_env.is_closed:
        eval_env.close()
    if not train_env.is_closed:
        train_env.close()


if __name__ == "__main__":
    main()
