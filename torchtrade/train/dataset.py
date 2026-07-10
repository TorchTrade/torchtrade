"""Turn historical bars into GRPO training rows {bar_index, prompt, system_prompt}."""
from torchtrade.train.prompts import build_training_prompt


def build_bar_dataset(actor, env, n_bars, system_prompt=None, user_prompt_fn=None):
    """One row per bar: the built (system, user) prompt + the bar_index used to score.

    `env.obs_at(i)` returns the deterministic observation for bar i (same access the
    reward oracle uses to seek to a bar). Kept minimal — no HF Dataset wrapping here;
    the trainer converts this list as needed.
    """
    rows = []
    for i in range(n_bars):
        td = env.obs_at(i)
        sysp, userp = build_training_prompt(actor, td, system_prompt, user_prompt_fn)
        rows.append({"bar_index": i, "prompt": userp, "system_prompt": sysp})
    return rows
