import pytest
from torchtrade.train.reward import TradingReward


class _FakeEnv:
    def score(self, bar_index, action):
        return float(action * 10 + bar_index)


def test_parses_answer_and_scores():
    r = TradingReward(env=_FakeEnv(), num_actions=3)
    assert r("<think>x</think><answer>2</answer>", bar_index=5) == pytest.approx(25.0)


@pytest.mark.parametrize("completion", ["no tag here", "<answer>9</answer>"], ids=["no-tag", "out-of-range"])
def test_unparseable_completion_defaults_action_zero_no_crash(completion):
    """No tag AND a well-formed-but-out-of-range index (reachable with constrain_actions=False)
    both fall back to action 0."""
    r = TradingReward(env=_FakeEnv(), num_actions=3)
    assert r(completion, bar_index=7) == pytest.approx(7.0)   # action 0 -> 0*10+7


def test_custom_reward_fn_overrides_scoring():
    seen = {}
    def rf(action, bar_index, env):
        seen.update(action=action, bar_index=bar_index)
        return 99.0
    r = TradingReward(env=_FakeEnv(), num_actions=3, reward_fn=rf)
    assert r("<answer>1</answer>", bar_index=3) == 99.0
    assert seen == {"action": 1, "bar_index": 3}
