"""Trading reward for GRPO: parse a completion's <answer>, score it at its bar."""
from torchtrade.actor.parsers import extract_action


class TradingReward:
    """Callable reward: completion text -> discrete action -> scalar reward.

    Default scoring is env.score(bar_index, action) (OneStepTradingEnv reward
    oracle). A custom reward_fn(action, bar_index, env) -> float overrides it.
    """

    def __init__(self, env, num_actions, reward_fn=None):
        self.env = env
        self.num_actions = num_actions
        self.reward_fn = reward_fn

    def __call__(self, completion: str, bar_index: int) -> float:
        action = extract_action(completion, num_actions=self.num_actions)
        if self.reward_fn is None:
            return float(self.env.score(bar_index, action))
        return float(self.reward_fn(action=action, bar_index=bar_index, env=self.env))
