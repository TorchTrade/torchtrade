from __future__ import annotations

import hydra
import pandas as pd
from torchtrade.actor.human import HumanActor
from torchtrade.envs.offline.infrastructure.utils import load_torch_trade_dataset


@hydra.main(config_path="", config_name="config", version_base="1.1")
def main(cfg: DictConfig):  # noqa: F821

    import tqdm
    from torchrl._utils import timeit
    from torchrl.record.loggers import generate_exp_name, get_logger
    from utils import make_environment, make_collector, log_metrics

    # Creante env
    df = load_torch_trade_dataset()
    train_df = df[(1440 * 21):]


    max_train_traj_length = cfg.collector.frames_per_batch // cfg.env.train_envs
    train_env = make_environment(
        train_df,
        cfg,
        train_num_envs=cfg.env.train_envs,
        max_train_traj_length=max_train_traj_length,
    )

    # Correct
    total_frames = cfg.collector.total_frames 

    market_data_keys = [key for key in train_env.observation_keys if key.startswith("market") ]
    account_state_key = [key for key in train_env.observation_keys if key.startswith("account") ]
    assert len(account_state_key) == 1

    features = train_env.sampler[0].get_feature_keys()
    # Create models (check utils_atari.py)
    actor = HumanActor(symbol=cfg.env.symbol,
                       features=features,
                       market_data_keys=market_data_keys,
                       account_state_key=account_state_key[0],
                       action_spec=train_env.action_spec)

    # Create collector
    collector = make_collector(
        cfg,
        train_env,
        actor,
    )

    # Create logger
    logger = None
    if cfg.logger.backend:
        exp_name = generate_exp_name("TorchTrade-online", cfg.logger.exp_name)
        logger = get_logger(
            cfg.logger.backend,
            logger_name="ppo_logging",
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

    collector_iter = iter(collector)
    total_iter = len(collector)
    for i in range(total_iter):
        timeit.printevery(1000, total_iter, erase=True)

        with timeit("collecting"):
            data = next(collector_iter)

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

        if logger is not None:
            time_dict = timeit.todict(prefix="time")
            metrics_to_log.update(timeit.todict(prefix="time"))
            metrics_to_log["time/speed"] = pbar.format_dict["rate"]
            metrics_to_log["time/SPS-collecting"] = frames_in_batch / time_dict["time/collecting"]
            metrics_to_log["time/SPS-total"] = frames_in_batch / sum(time_dict.values())
            log_metrics(logger, metrics_to_log, collected_frames)

        collector.update_policy_weights_()

    collector.shutdown()
    if not train_env.is_closed:
        train_env.close()


if __name__ == "__main__":
    main()