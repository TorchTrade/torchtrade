"""Tests for PolyTimeBarEnv."""

import numpy as np
import pytest
import torch
from tensordict import TensorDict
from torchrl.data import Bounded

from tests.envs.polymarket.mocks import MockPolymarketObserver, MockPolymarketTrader


class TestPolyTimeBarEnv:
    """Tests for PolyTimeBarEnv."""

    @pytest.fixture
    def env(self):
        from torchtrade.envs.live.polymarket.env import (
            PolyTimeBarEnv,
            PolyTimeBarEnvConfig,
        )

        config = PolyTimeBarEnvConfig(
            yes_token_id="tok_yes_1",
            execute_on="1Hour",
            action_levels=[-1, 0, 1],
            initial_cash=10_000.0,
        )
        observer = MockPolymarketObserver(yes_price=0.72)
        trader = MockPolymarketTrader(initial_balance=10_000.0, yes_price=0.72)
        env = PolyTimeBarEnv(
            config=config,
            observer=observer,
            trader=trader,
        )
        env._wait_for_next_timestamp = lambda: None  # Skip sleep in tests
        return env

    def test_observation_spec_has_required_keys(self, env):
        """observation_spec includes market_state and account_state."""
        spec = env.observation_spec
        assert "market_state" in spec.keys()
        assert "account_state" in spec.keys()

    def test_action_spec(self, env):
        """Action spec matches action_levels count."""
        assert env.action_spec.n == 3

    def test_reset_returns_valid_tensordict(self, env):
        """reset() returns TensorDict with market_state and account_state."""
        td = env.reset()
        assert "market_state" in td.keys()
        assert "account_state" in td.keys()
        assert td["market_state"].shape == (5,)
        assert td["account_state"].shape == (6,)

    def test_account_state_initial_values(self, env):
        """After reset, account state shows flat position."""
        td = env.reset()
        acct = td["account_state"]
        assert acct[0].item() == pytest.approx(0.0)  # exposure_pct
        assert acct[1].item() == pytest.approx(0.0)  # position_direction
        assert acct[4].item() == pytest.approx(1.0)  # leverage
        assert acct[5].item() == pytest.approx(1.0)  # distance_to_liquidation

    @pytest.mark.parametrize(
        "action_idx,expected_direction",
        [
            (0, -1),  # action_levels[-1] -> short/NO
            (1, 0),   # action_levels[0]  -> flat
            (2, 1),   # action_levels[1]  -> long/YES
        ],
        ids=["buy-no", "flat", "buy-yes"],
    )
    def test_step_action_direction(self, env, action_idx, expected_direction):
        """Step with different actions produces correct position direction."""
        env.reset()
        td_in = TensorDict({"action": torch.tensor(action_idx)}, batch_size=())
        td_out = env._step(td_in)
        assert "reward" in td_out.keys()
        assert "done" in td_out.keys()
        assert "terminated" in td_out.keys()
        # Check position direction matches
        direction = td_out["account_state"][1].item()
        if expected_direction != 0:
            assert direction == pytest.approx(expected_direction, abs=0.1)
        else:
            assert direction == pytest.approx(0.0, abs=0.1)

    def test_step_flat_no_trade(self, env):
        """action=flat when already flat produces no trade."""
        env.reset()
        initial_balance = env.trader.get_balance()
        td_in = TensorDict({"action": torch.tensor(1)}, batch_size=())  # 0 -> flat
        env._step(td_in)
        assert env.trader.get_balance() == pytest.approx(initial_balance)

    def test_termination_on_market_close(self, env):
        """Episode terminates when market resolves."""
        env.reset()
        env.observer.market_closed = True
        td_in = TensorDict({"action": torch.tensor(1)}, batch_size=())
        td_out = env._step(td_in)
        assert td_out["terminated"].item() is True

    def test_supplementary_observers_merged(self):
        """Supplementary observer specs and observations are merged."""
        from torchtrade.envs.live.polymarket.env import (
            PolyTimeBarEnv,
            PolyTimeBarEnvConfig,
        )

        config = PolyTimeBarEnvConfig(
            yes_token_id="tok_yes_1",
            action_levels=[-1, 0, 1],
        )

        # Create a mock supplementary observer
        class FakeSupplementary:
            def get_observation_spec(self):
                return {
                    "extra_data": Bounded(
                        low=-torch.inf, high=torch.inf, shape=(3,), dtype=torch.float32
                    )
                }

            def get_observations(self):
                return {"extra_data": np.array([1.0, 2.0, 3.0], dtype=np.float32)}

        env = PolyTimeBarEnv(
            config=config,
            observer=MockPolymarketObserver(),
            trader=MockPolymarketTrader(),
            supplementary_observers=[FakeSupplementary()],
        )
        env._wait_for_next_timestamp = lambda: None
        assert "extra_data" in env.observation_spec.keys()
        td = env.reset()
        assert "extra_data" in td.keys()
        assert td["extra_data"].shape == (3,)

    def test_supplementary_key_collision_raises(self):
        """Key collision between supplementary observers raises ValueError."""
        from torchtrade.envs.live.polymarket.env import (
            PolyTimeBarEnv,
            PolyTimeBarEnvConfig,
        )

        config = PolyTimeBarEnvConfig(
            yes_token_id="tok_yes_1",
            action_levels=[-1, 0, 1],
        )

        class CollidingObserver:
            def get_observation_spec(self):
                return {
                    "market_state": Bounded(
                        low=0, high=1, shape=(5,), dtype=torch.float32
                    )
                }

            def get_observations(self):
                return {"market_state": np.zeros(5, dtype=np.float32)}

        with pytest.raises(ValueError, match="collision"):
            PolyTimeBarEnv(
                config=config,
                observer=MockPolymarketObserver(),
                trader=MockPolymarketTrader(),
                supplementary_observers=[CollidingObserver()],
            )

    def test_max_steps_truncation(self):
        """Episode truncates after max_steps."""
        from torchtrade.envs.live.polymarket.env import (
            PolyTimeBarEnv,
            PolyTimeBarEnvConfig,
        )

        config = PolyTimeBarEnvConfig(
            yes_token_id="tok_yes_1",
            action_levels=[-1, 0, 1],
            max_steps=2,
        )
        env = PolyTimeBarEnv(
            config=config,
            observer=MockPolymarketObserver(),
            trader=MockPolymarketTrader(),
        )
        env._wait_for_next_timestamp = lambda: None
        env.reset()
        td1 = env._step(TensorDict({"action": torch.tensor(1)}, batch_size=()))
        assert td1["truncated"].item() is False
        td2 = env._step(TensorDict({"action": torch.tensor(1)}, batch_size=()))
        assert td2["truncated"].item() is True

    def test_bankruptcy_termination(self):
        """Episode terminates when balance drops below threshold."""
        from torchtrade.envs.live.polymarket.env import (
            PolyTimeBarEnv,
            PolyTimeBarEnvConfig,
        )

        config = PolyTimeBarEnvConfig(
            yes_token_id="tok_yes_1",
            action_levels=[-1, 0, 1],
            done_on_bankruptcy=True,
            bankrupt_threshold=0.5,
        )
        # Start with very low balance to trigger bankruptcy
        trader = MockPolymarketTrader(initial_balance=1.0, yes_price=0.72)
        env = PolyTimeBarEnv(
            config=config,
            observer=MockPolymarketObserver(),
            trader=trader,
        )
        env._wait_for_next_timestamp = lambda: None
        env.reset()
        env._initial_balance = 10_000.0  # Pretend we started rich
        td = env._step(TensorDict({"action": torch.tensor(1)}, batch_size=()))
        assert td["terminated"].item() is True

    def test_missing_market_id_raises(self):
        """ValueError raised when no market identifier is provided."""
        from torchtrade.envs.live.polymarket.env import (
            PolyTimeBarEnv,
            PolyTimeBarEnvConfig,
        )

        config = PolyTimeBarEnvConfig()  # All empty
        with pytest.raises(ValueError, match="market identifier"):
            PolyTimeBarEnv(
                config=config,
                observer=MockPolymarketObserver(),
                trader=MockPolymarketTrader(),
            )

    def test_close_position_on_reset(self):
        """When close_position_on_reset=True, positions are closed on reset."""
        from torchtrade.envs.live.polymarket.env import (
            PolyTimeBarEnv,
            PolyTimeBarEnvConfig,
        )

        config = PolyTimeBarEnvConfig(
            yes_token_id="tok_yes_1",
            action_levels=[-1, 0, 1],
            close_position_on_reset=True,
            close_position_on_init=False,
        )
        trader = MockPolymarketTrader(initial_balance=10_000.0)
        env = PolyTimeBarEnv(
            config=config,
            observer=MockPolymarketObserver(),
            trader=trader,
        )
        env._wait_for_next_timestamp = lambda: None
        env.reset()
        # Buy YES
        env._step(TensorDict({"action": torch.tensor(2)}, batch_size=()))
        # Reset should close position
        env.reset()
        assert env.position.current_position == 0.0
