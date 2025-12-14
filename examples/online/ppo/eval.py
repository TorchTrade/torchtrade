from __future__ import annotations

import warnings

import hydra
from sympy.logic.boolalg import true
from torchrl._utils import compile_with_warmup
import datasets


@hydra.main(config_path="", config_name="config", version_base="1.1")
def main(cfg: DictConfig):  # noqa: F821
    import matplotlib.pyplot as plt
    import torch
    from torchrl._utils import timeit
    from torchrl.envs import ExplorationType, set_exploration_type
    from utils import make_environment, make_ppo_models

    torch.set_float32_matmul_precision("high")
    torch.manual_seed(cfg.env.seed)

    device = cfg.optim.device
    if device in ("", None):
        if torch.cuda.is_available():
            device = "cuda:0"
        else:
            device = "cpu"
    device = torch.device(device)
    print("USING DEVICE: ", device)

    # Creante env
    df = datasets.load_dataset(cfg.env.data_path)
    df = df["train"].to_pandas()
    total_data_1min = len(df)
    test_df = df[0:(1440 *21)] # 14 days
    train_df = df[(1440 * 21):]

    # execution on 15 min -> 2016 = 21 tage
    rollout_length = 500

    train_env, eval_env = make_environment(
        train_df,
        test_df,
        cfg,
        train_num_envs=4,
        eval_num_envs=2,
        max_train_traj_length=rollout_length,
        max_eval_traj_length=rollout_length
    )
    eval_env.to(device)
    train_env.to(device)
    
    test_env = eval_env

    actor, _ = make_ppo_models(
        eval_env,
        device=device,
        cfg=cfg,
    )

    #actor.load_state_dict(torch.load("../../../"+cfg.eval.model_path))

    # Get test rewards
    with torch.no_grad(), set_exploration_type(
        ExplorationType.DETERMINISTIC
    ), timeit("eval"):

        actor.eval()
        eval_rollout = test_env.rollout(
            rollout_length,
            actor,
            auto_cast_to_device=True,
            break_when_any_done=True,
            trust_policy=True
        )
        eval_rollout.squeeze()
        eval_reward = eval_rollout["next", "reward"].sum(-2)
        # for i, r in enumerate(eval_reward):
        #     print(f"Eval reward {i}: {r.item()}")

        initial_portfolio_values = eval_rollout["account_state"][:, 0, 0]
        final_portfolio_values = eval_rollout["account_state"][:, -1, 0]
        returns = (final_portfolio_values-initial_portfolio_values)/initial_portfolio_values
        for i, r in enumerate(returns):
            print(f"Total return {i}: {r.item()*100:.2f}%")

        fig = test_env.base_env.render_history(return_fig=True)
        plt.show()
        #eval_env.reset()
        # TODO: add metric like daily profit %
        # metrics_to_log["eval/daily_profit_pct"] = 



    if not eval_env.is_closed:
        eval_env.close()



if __name__ == "__main__":
    main()