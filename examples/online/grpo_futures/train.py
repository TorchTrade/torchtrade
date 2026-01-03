"""GRPO training example for FuturesOneStepEnv.

This script demonstrates GRPO training on the FuturesOneStepEnv environment
for futures trading with leverage, stop-loss, and take-profit support.
"""
from __future__ import annotations

import warnings

import hydra
from torchrl._utils import compile_with_warmup
import datasets
from torchtrade.losses import GRPOLoss


@hydra.main(config_path="", config_name="config", version_base="1.1")
def main(cfg: DictConfig):  # noqa: F821

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
    print("USING DEVICE: ", device)

    # Load data from HuggingFace
    df = datasets.load_dataset(cfg.env.data_path)
    df = df["train"].to_pandas()

    train_df = df[df['0'] < cfg.env.test_split_start]
    test_df  = df[df['0'] >= cfg.env.test_split_start]

    print("len train", len(train_df))
    print("len test", len(test_df))

    max_train_traj_length = cfg.collector.frames_per_batch // cfg.env.train_envs
    # TODO: possibly wrong as the test_df is in 1min freq
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
        exp_name = generate_exp_name("TorchTrade-FuturesOneStep-GRPO", cfg.logger.exp_name)
        logger = get_logger(
            cfg.logger.backend,
            logger_name="grpo_futures_logging",
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
        optim.zero_grad(set_to_none=True)
        # PERF: batch is already on device from collector, skip redundant transfer

        # Forward pass GRPO loss
        loss_td = loss_module(batch)
        loss = loss_td["loss_objective"] + loss_td["loss_entropy"]
        # Backward pass
        loss.backward()

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
        update = CudaGraphModule(update, in_keys=[], out_keys=[], warmup=5)

    collector_iter = iter(collector)
    total_iter = len(collector)
    for i in range(total_iter):
        timeit.printevery(1000, total_iter, erase=True)

        with timeit("collecting"):
            # PERF: Use non_blocking=True for async transfer
            data = next(collector_iter).to(device, non_blocking=True)

        metrics_to_log = {}
        frames_in_batch = data.numel()
        collected_frames += frames_in_batch
        pbar.update(frames_in_batch)

        # Get training rewards and episode lengths
        batch_reward = data["next", "reward"].mean()
        # PERF: Defer .item() calls to reduce GPU sync points
        # These will be logged later, so we compute them but don't block
        action_std = data["action"].float().std()
        metrics_to_log.update({"train/reward": batch_reward, "train/action_std": action_std})

        with timeit("training"):
            with timeit("update"):
                torch.compiler.cudagraph_mark_step_begin()
                loss = update(data)
            loss = loss.clone()

        # Get training losses and times
        for key, value in loss.items():
            metrics_to_log.update({f"train/{key}": value.item()})

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
                # Render history (available in SeqFuturesSLTPEnv for eval)
                fig = eval_env.base_env.render_history(return_fig=True)
                eval_env.reset()
                if fig is not None and logger is not None:
                    metrics_to_log["eval/history"] = wandb.Image(fig[0])
                actor.train()

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
