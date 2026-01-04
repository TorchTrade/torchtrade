"""PPO training example for SeqFuturesEnv.

This script demonstrates PPO training on the SeqFuturesEnv environment
for futures trading with leverage support.
"""
from __future__ import annotations

import warnings

import hydra
from torchrl._utils import compile_with_warmup
import datasets


@hydra.main(config_path="", config_name="config", version_base="1.1")
def main(cfg: DictConfig):  # noqa: F821

    import torch.optim
    import tqdm
    import wandb
    from tensordict import TensorDict
    from tensordict.nn import CudaGraphModule

    from torchrl._utils import timeit
    from torchrl.data import LazyTensorStorage, TensorDictReplayBuffer
    from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
    from torchrl.envs import ExplorationType, set_exploration_type
    from torchrl.objectives import ClipPPOLoss
    from torchrl.objectives.value.advantages import GAE
    from torchrl.record.loggers import generate_exp_name, get_logger
    from utils import make_environment, make_ppo_models, make_collector, log_metrics
    from torchtrade.envs.transforms import CoverageTracker

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
    train_df = df[df['0'] < cfg.env.test_split_start]
    test_df  = df[df['0'] >= cfg.env.test_split_start]

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

    # Extract CoverageTracker for logging (if available)
    coverage_tracker = None
    for transform in train_env.transform:
        if isinstance(transform, CoverageTracker):
            coverage_tracker = transform
            break

    total_frames = cfg.collector.total_frames
    frames_per_batch = cfg.collector.frames_per_batch
    mini_batch_size = cfg.loss.mini_batch_size
    test_interval = cfg.logger.test_interval

    compile_mode = None
    if cfg.compile.compile:
        compile_mode = cfg.compile.compile_mode
        if compile_mode in ("", None):
            if cfg.compile.cudagraphs:
                compile_mode = "default"
            else:
                compile_mode = "reduce-overhead"

    # Create models
    actor, critic = make_ppo_models(
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

    # Create data buffer
    sampler = SamplerWithoutReplacement(drop_last=True)
    data_buffer = TensorDictReplayBuffer(
        storage=LazyTensorStorage(
            frames_per_batch, compilable=cfg.compile.compile, device=device
        ),
        sampler=sampler,
        batch_size=mini_batch_size,
        compilable=cfg.compile.compile,
    )

    # Create loss and advantage modules
    adv_module = GAE(
        gamma=cfg.loss.gamma,
        lmbda=cfg.loss.gae_lambda,
        value_network=critic,
        average_gae=False,
        device=device,
        time_dim=-3,
        vectorized=not cfg.compile.compile,
    )
    loss_module = ClipPPOLoss(
        actor_network=actor,
        critic_network=critic,
        clip_epsilon=cfg.loss.clip_epsilon,
        loss_critic_type=cfg.loss.loss_critic_type,
        entropy_coef=cfg.loss.entropy_coef,
        critic_coef=cfg.loss.critic_coef,
        normalize_advantage=True,
    )

    # Create optimizer
    optim = torch.optim.Adam(
        loss_module.parameters(),
        lr=cfg.optim.lr,
        weight_decay=cfg.optim.weight_decay,
        eps=cfg.optim.eps,
    )

    # Create logger
    logger = None
    if cfg.logger.backend:
        exp_name = generate_exp_name("TorchTrade-Futures-PPO", cfg.logger.exp_name)
        logger = get_logger(
            cfg.logger.backend,
            logger_name="ppo_futures_logging",
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
    num_network_updates = torch.zeros((), dtype=torch.int64, device=device)
    pbar = tqdm.tqdm(total=total_frames)
    num_mini_batches = frames_per_batch // mini_batch_size
    total_network_updates = (
        (total_frames // frames_per_batch) * cfg.loss.ppo_epochs * num_mini_batches
    )

    def update(batch, num_network_updates):
        optim.zero_grad(set_to_none=True)
        alpha = torch.ones((), device=device)
        if cfg_optim_anneal_lr:
            alpha = 1 - (num_network_updates / total_network_updates)
            for group in optim.param_groups:
                group["lr"] = cfg_optim_lr * alpha
        if cfg_loss_anneal_clip_eps:
            loss_module.clip_epsilon.copy_(cfg_loss_clip_epsilon * alpha)

        batch = batch.to(device, non_blocking=True)

        # Forward pass PPO loss
        loss = loss_module(batch)
        loss_sum = loss["loss_critic"] + loss["loss_objective"] + loss["loss_entropy"]

        # Backward pass
        loss_sum.backward()
        torch.nn.utils.clip_grad_norm_(
            loss_module.parameters(), max_norm=cfg_optim_max_grad_norm
        )

        # Update the networks
        optim.step()
        return loss.detach().set("alpha", alpha), num_network_updates

    if cfg.compile.compile:
        update = compile_with_warmup(update, mode=compile_mode, warmup=1)
        adv_module = compile_with_warmup(adv_module, mode=compile_mode, warmup=1)

    if cfg.compile.cudagraphs:
        warnings.warn(
            "CudaGraphModule is experimental and may lead to silently wrong results. Use with caution.",
            category=UserWarning,
        )
        update = CudaGraphModule(update, in_keys=[], out_keys=[], warmup=5)
        adv_module = CudaGraphModule(adv_module)

    # Extract cfg variables
    cfg_loss_ppo_epochs = cfg.loss.ppo_epochs
    cfg_optim_anneal_lr = cfg.optim.anneal_lr
    cfg_optim_lr = cfg.optim.lr
    cfg_loss_anneal_clip_eps = cfg.loss.anneal_clip_epsilon
    cfg_loss_clip_epsilon = cfg.loss.clip_epsilon
    cfg_optim_max_grad_norm = cfg.optim.max_grad_norm
    cfg.loss.clip_epsilon = cfg_loss_clip_epsilon
    losses = TensorDict(batch_size=[cfg_loss_ppo_epochs, num_mini_batches])

    collector_iter = iter(collector)
    total_iter = len(collector)
    for i in range(total_iter):
        timeit.printevery(1000, total_iter, erase=True)

        with timeit("collecting"):
            data = next(collector_iter).to(device)

        metrics_to_log = {}
        frames_in_batch = data.numel()
        metrics_to_log["train/action_std"] = data["action"].float().std().item()
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

        with timeit("training"):
            for j in range(cfg_loss_ppo_epochs):
                # Compute GAE
                with torch.no_grad(), timeit("adv"):
                    torch.compiler.cudagraph_mark_step_begin()
                    data = adv_module(data)
                    if compile_mode:
                        data = data.clone()
                with timeit("rb - extend"):
                    # Update the data buffer
                    data_reshape = data.reshape(-1)
                    data_buffer.extend(data_reshape)

                for k, batch in enumerate(data_buffer):
                    with timeit("update"):
                        torch.compiler.cudagraph_mark_step_begin()
                        loss, num_network_updates = update(
                            batch, num_network_updates=num_network_updates
                        )
                    loss = loss.clone()
                    num_network_updates = num_network_updates.clone()
                    losses[j, k] = loss.select(
                        "loss_critic", "loss_entropy", "loss_objective"
                    )

        # Get training losses and times
        losses_mean = losses.apply(lambda x: x.float().mean(), batch_size=[])
        for key, value in losses_mean.items():
            metrics_to_log.update({f"train/{key}": value.item()})
        metrics_to_log.update(
            {
                "train/lr": loss["alpha"] * cfg_optim_lr,
                "train/clip_epsilon": loss["alpha"] * cfg_loss_clip_epsilon,
            }
        )

        # Evaluation
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

                # Render history - this works because base_env provides a render_history method
                # that delegates to the underlying environment
                fig = eval_env.base_env.render_history(return_fig=True)
                eval_env.reset()
                if fig is not None and logger is not None:
                    # render_history returns a figure directly for SeqFuturesEnv
                    metrics_to_log["eval/history"] = wandb.Image(fig[0])
                torch.save(actor.state_dict(), f"ppo_futures_policy_{i}.pth")
                actor.train()

        # Log coverage metrics (if available and enabled)
        if coverage_tracker is not None:
            coverage_stats = coverage_tracker.get_coverage_stats()
            if coverage_stats["enabled"]:
                metrics_to_log["coverage/coverage"] = coverage_stats["coverage"]
                metrics_to_log["coverage/visited"] = coverage_stats["visited_positions"]
                metrics_to_log["coverage/unvisited"] = coverage_stats["unvisited_positions"]
                metrics_to_log["coverage/entropy"] = coverage_stats["coverage_entropy"]
                metrics_to_log["coverage/mean_visits"] = coverage_stats["mean_visits_per_position"]
                metrics_to_log["coverage/std_visits"] = coverage_stats["std_visits"]

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
