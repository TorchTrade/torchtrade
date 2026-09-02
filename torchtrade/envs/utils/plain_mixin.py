"""Shared behaviour for the plain (non-bracket) futures trading environments."""

from torchrl.data import Categorical

from torchtrade.envs.core.default_rewards import log_return_reward


class PlainFuturesLiveEnv:
    """What the four plain futures envs build that the four SLTP ones do not.

    A mixin for the same reason `SLTPMixin` is one: the four SLTP siblings inherit the
    same `<venue>BaseTorchTradingEnv`, size brackets from `action_map`, and their config
    has no `action_levels` -- so this cannot sit on `TorchTradeFuturesLiveEnv` without
    that class lying to anyone reading the SLTP MRO. It cannot be an `__init__` either:
    the venues pass different credentials to `super()`.

    Folded for the drift, not the line count. #425 was these two lines in four copies:
    binance derived its levels from a helper while three venues hard-coded a five-level
    list, so one unset config gave a 3-action space on one venue and 5 on the others --
    and `action_spec.n` is what a checkpoint binds to.
    """

    def _init_plain_trading(self, config, reward_function=None):
        """Call from the leaf `__init__`, after `super().__init__(...)`. Ordering is not
        delicate: torchrl's `_EnvPostInit` runs the whole chain before reading specs."""
        self.reward_function = reward_function or log_return_reward
        self.action_levels = config.action_levels
        self.action_spec = Categorical(len(self.action_levels))
