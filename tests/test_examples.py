"""
Tests for training examples.

This module tests that training examples run without errors using:
1. Mock environments for online (Alpaca) examples
2. Synthetic data for offline examples
3. Minimal training parameters for quick validation

Similar to TorchRL's sota-tests approach.
"""

import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import numpy as np

# Get the repository root
REPO_ROOT = Path(__file__).parent.parent

# HuggingFace dataset path for market OHLCV data (used by online examples)
HF_MARKET_DATA_PATH = "Torch-Trade/btcusdt_spot_1m_03_2023_to_12_2025"


# =============================================================================
# Online Example Tests (using mocks)
# =============================================================================

class TestOnlineExamplesWithMocks:
    """Test online examples using mock Alpaca environment."""

    def test_alpaca_env_with_mocks(self):
        """Test that AlpacaTorchTradingEnv works with mocks."""
        from torchtrade.envs.live.alpaca.env import (
            AlpacaTorchTradingEnv,
            AlpacaTradingEnvConfig,
        )
        import sys
        sys.path.insert(0, str(REPO_ROOT))
        from tests.envs.alpaca.mocks import MockObserver, MockTrader

        config = AlpacaTradingEnvConfig(
            symbol="BTC/USD",
            window_sizes=[10],
        )

        mock_observer = MockObserver(window_sizes=[10])
        mock_trader = MockTrader(initial_cash=10000.0)

        env = AlpacaTorchTradingEnv(
            config=config,
            observer=mock_observer,
            trader=mock_trader,
        )

        # Skip wait delays
        env._wait_for_next_timestamp = lambda: None

        # Test reset
        td = env.reset()
        assert td is not None

        # Test multiple steps
        for _ in range(10):
            action = torch.tensor(np.random.randint(0, 3))
            td = env._step(td.set("action", action))
            assert "reward" in td.keys()
            assert "done" in td.keys()

        env.close()

    def test_mock_environment_rollout(self):
        """Test running a rollout with mocked environment."""
        from torchtrade.envs.live.alpaca.env import (
            AlpacaTorchTradingEnv,
            AlpacaTradingEnvConfig,
        )
        import sys
        sys.path.insert(0, str(REPO_ROOT))
        from tests.envs.alpaca.mocks import MockObserver, MockTrader
        from tensordict.nn import TensorDictModule
        from torch import nn

        config = AlpacaTradingEnvConfig(
            symbol="BTC/USD",
            window_sizes=[10],
        )

        mock_observer = MockObserver(window_sizes=[10], num_features=4)
        mock_trader = MockTrader(initial_cash=10000.0)

        env = AlpacaTorchTradingEnv(
            config=config,
            observer=mock_observer,
            trader=mock_trader,
        )
        env._wait_for_next_timestamp = lambda: None

        # Create a simple random policy
        class RandomPolicy(nn.Module):
            def __init__(self, n_actions):
                super().__init__()
                self.n_actions = n_actions

            def forward(self, x):
                batch_size = x.shape[0] if x.dim() > 1 else 1
                return torch.randint(0, self.n_actions, (batch_size,))

        policy = TensorDictModule(
            RandomPolicy(3),
            in_keys=["account_state"],
            out_keys=["action"],
        )

        # Run a short rollout
        td = env.reset()
        rewards = []
        for _ in range(5):
            td = policy(td)
            td = env._step(td)
            rewards.append(td["reward"].item())

        assert len(rewards) == 5
        env.close()


# =============================================================================
# Offline data plumbing (dataset_to_td + replay buffer)
# =============================================================================

def _check_hf_market_data_available():
    """Check if HuggingFace market data dataset is accessible."""
    import os
    import warnings
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    try:
        # Metadata only -- load_dataset here would pull ~1.4M rows at collection time
        # just to answer a yes/no.
        from huggingface_hub import HfApi
        HfApi().dataset_info(HF_MARKET_DATA_PATH, token=token)
        return True
    except Exception as e:
        # Use warnings to make debug info visible in pytest output
        warnings.warn(
            f"HF market data '{HF_MARKET_DATA_PATH}' check failed "
            f"(token={'set' if token else 'NOT SET'}): {e}",
            UserWarning
        )
        return False


_HF_MARKET_DATA_AVAILABLE = None


def hf_market_data_available():
    """Cached check for HuggingFace market data availability."""
    global _HF_MARKET_DATA_AVAILABLE
    if _HF_MARKET_DATA_AVAILABLE is None:
        _HF_MARKET_DATA_AVAILABLE = _check_hf_market_data_available()
    return _HF_MARKET_DATA_AVAILABLE


@pytest.fixture
def hf_dataset():
    """An ordinary offline-RL transition dataset: dotted columns for the `next` subtd,
    and no `next.terminated` (many real datasets only record `done`)."""
    from datasets import Dataset
    n, window, obs_dim = 16, 12, 4
    rng = np.random.default_rng(0)
    return Dataset.from_dict({
        "observation": rng.standard_normal((n, window, obs_dim)).tolist(),
        "action": rng.integers(0, 3, n).tolist(),
        "next.observation": rng.standard_normal((n, window, obs_dim)).tolist(),
        "next.reward": rng.standard_normal(n).tolist(),
        "next.done": [False] * n,
    })


def test_dotted_columns_become_nested_keys(hf_dataset):
    """dataset_to_td's one non-trivial job: 'next.reward' must become
    ('next', 'reward'), not a literal top-level key."""
    from torchtrade.utils import dataset_to_td

    td = dataset_to_td(hf_dataset)

    assert td.batch_size == torch.Size([16])
    assert td["next", "reward"].shape == (16,)
    assert td["next", "observation"].shape == (16, 12, 4)
    assert td["observation"].shape == (16, 12, 4)
    assert "next.reward" not in td.keys()


@pytest.fixture
def iql_offline_utils():
    # Loaded by path under a unique name: 8 examples ship a module called `utils` and
    # examples/ has no __init__.py, so a plain import would poison sys.modules.
    import importlib.util

    path = REPO_ROOT / "examples" / "offline_rl" / "iql" / "utils.py"
    spec = importlib.util.spec_from_file_location("_iql_offline_utils", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("source_shape", [(8,), (8, 1)], ids=["flat", "already-2d"])
def test_offline_buffer_gives_reward_done_terminated_a_trailing_dim(
    tmp_path, source_shape, iql_offline_utils
):
    """Regression: DiscreteIQLLoss needs (*batch, 1) to match state_value and raises
    'All input tensors ... must share a unique shape' on flat (*batch,). dataset_to_td
    and tensordict.load both yield flat rewards, so make_offline_replay_buffer must add
    the dim -- and must not add a second one to sources that already have it."""
    from tensordict import TensorDict

    n = 8
    TensorDict({
        "observation": torch.randn(n, 3),
        "action": torch.randint(0, 3, (n,)),
        "next": TensorDict({
            "observation": torch.randn(n, 3),
            "reward": torch.randn(*source_shape),
            "done": torch.zeros(*source_shape, dtype=torch.bool),
            "terminated": torch.zeros(*source_shape, dtype=torch.bool),
        }, batch_size=[n]),
    }, batch_size=[n]).save(str(tmp_path))

    rb_cfg = SimpleNamespace(data_path=str(tmp_path), buffer_size=n, batch_size=4)
    buffer = iql_offline_utils.make_offline_replay_buffer(rb_cfg, env=None)

    sample = buffer.sample()
    for key in (("next", "reward"), ("next", "done"), ("next", "terminated")):
        assert sample[key].shape == (4, 1), f"{key}: {sample[key].shape}"


def test_offline_buffer_hf_branch_synthesises_missing_terminated(
    monkeypatch, hf_dataset, iql_offline_utils
):
    """A replay dataset with next.done but no next.terminated is ordinary, and the loss
    still needs terminated. Reaching into td.get() for an absent key returns None, so
    this branch used to die with AttributeError before it ever got to the buffer."""
    import datasets

    monkeypatch.setattr(datasets, "load_dataset", lambda *a, **kw: hf_dataset)

    rb_cfg = SimpleNamespace(data_path="Some-Org/some-transitions", buffer_size=16, batch_size=4)
    buffer = iql_offline_utils.make_offline_replay_buffer(rb_cfg, env=None)

    sample = buffer.sample()
    assert sample["next", "terminated"].shape == (4, 1)
    assert torch.equal(sample["next", "terminated"], sample["next", "done"])


def test_run_command_uses_the_interpreter_running_the_tests():
    """A stale `python` on PATH would smoke-test the examples against a different
    torchrl than the pinned one, and pass."""
    assert run_command(
        'python -c "import sys; sys.exit(0 if sys.executable == %r else 1)"' % sys.executable
    ) == 0


# =============================================================================
# SOTA-Style Example Tests (subprocess execution)
# =============================================================================

# Commands to run examples with minimal parameters
# All examples now use HuggingFace datasets for market data
# NOTE: Must use env.train_envs>=2 and env.eval_envs>=2 to avoid batch dimension squeeze issues
EXAMPLE_COMMANDS = {
    # ==========================================================================
    # IQL Examples
    # ==========================================================================

    "iql_online": (
        "python examples/online_rl/iql/train.py "
        "collector.total_frames=10 "
        "collector.frames_per_batch=5 "
        "collector.init_random_frames=5 "
        "env.train_envs=2 "
        "replay_buffer.batch_size=5 "
        "replay_buffer.buffer_size=20 "
        "logger.backend= "
        "logger.eval_iter=1000000 "
        "env.test_split_start=2025-07-01 "
    ),

    "iql_offline": (
        "python examples/offline_rl/iql/train.py "
        "optim.gradient_steps=5 "
        "optim.device=cpu "
        "replay_buffer.data_path=synthetic "
        "replay_buffer.batch_size=16 "
        "replay_buffer.buffer_size=50 "
        "logger.backend= "
        "logger.eval_iter=1000000 "
        "logger.eval_steps=5 "
    ),

    # ==========================================================================
    # DSAC Example
    # ==========================================================================

    "dsac_online": (
        "python examples/online_rl/dsac/train.py "
        "collector.total_frames=10 "
        "collector.frames_per_batch=5 "
        "collector.init_random_frames=5 "
        "env.train_envs=2 "
        "env.eval_envs=2 "
        "optim.batch_size=5 "
        "replay_buffer.size=20 "
        "logger.backend= "
        "logger.eval_iter=1000000 "
        "env.test_split_start=2025-07-01 "
    ),

    # ==========================================================================
    # PPO Example
    # ==========================================================================

    "ppo_online": (
        "python examples/online_rl/ppo/train.py "
        "collector.total_frames=10 "
        "collector.frames_per_batch=10 "
        "loss.mini_batch_size=5 "
        "env.train_envs=1 "
        "logger.backend= "
        "logger.test_interval=1000000 "
        "env.test_split_start=2025-07-01 "
    ),

    # ==========================================================================
    # PPO + Chronos Example (requires optional chronos-forecasting package)
    # ==========================================================================

    # TODO: Enable ppo_chronos test once chronos-forecasting is added as optional dependency
    # Requires: pip install git+https://github.com/amazon-science/chronos-forecasting.git
    # "ppo_chronos_online": (
    #     "python examples/online_rl/ppo_chronos/train.py "
    #     "collector.total_frames=10 "
    #     "collector.frames_per_batch=10 "
    #     "loss.mini_batch_size=5 "
    #     "env.train_envs=2 "
    #     "logger.backend= "
    #     "logger.test_interval=1000000 "
    #     "env.test_split_start=2025-07-01 "
    # ),

    # ==========================================================================
    # GRPO Example
    # ==========================================================================

    "grpo_online": (
        "python examples/online_rl/grpo/train.py "
        "collector.total_frames=10 "
        "collector.frames_per_batch=10 "
        "env.train_envs=2 "
        "logger.backend= "
        "logger.test_interval=1000000 "
        "env.test_split_start=2025-07-01 "
    ),

    # ==========================================================================
    # DQN Example
    # ==========================================================================

    "dqn_online": (
        "python examples/online_rl/dqn/train.py "
        "collector.total_frames=10 "
        "collector.frames_per_batch=5 "
        "collector.init_random_frames=5 "
        "env.train_envs=2 "
        "env.eval_envs=2 "
        "buffer.batch_size=5 "
        "buffer.buffer_size=20 "
        "logger.backend= "
        "logger.test_interval=1000000 "
        "env.test_split_start=2025-07-01 "
    ),
}


def run_command(command: str, timeout: int = 300) -> int:
    """
    Run a shell command and return the exit code.

    Args:
        command: The command to run
        timeout: Timeout in seconds

    Returns:
        Exit code (0 for success)
    """
    env = os.environ.copy()
    env["WANDB_MODE"] = "disabled"  # Disable wandb logging

    # Use the interpreter running the tests; a stale `python` on PATH would smoke-test
    # the examples against a different torchrl than the one under test.
    if command.startswith("python "):
        command = sys.executable + command[len("python"):]

    process = subprocess.Popen(
        command,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        cwd=str(REPO_ROOT),
        env=env,
    )

    try:
        stdout, _ = process.communicate(timeout=timeout)
        if process.returncode != 0:
            print(f"Command failed with exit code {process.returncode}")
            print(stdout.decode() if stdout else "")
        return process.returncode
    except subprocess.TimeoutExpired:
        process.kill()
        raise


@pytest.mark.skipif(
    len(EXAMPLE_COMMANDS) == 0,
    reason="No example commands configured yet"
)
@pytest.mark.skipif(
    not hf_market_data_available(),
    reason=f"HuggingFace market data '{HF_MARKET_DATA_PATH}' not accessible (may require auth)"
)
@pytest.mark.parametrize("name,command", list(EXAMPLE_COMMANDS.items()))
def test_example_commands(name: str, command: str):
    """Run example training scripts with minimal parameters."""
    returncode = run_command(command, timeout=300)
    assert returncode == 0, f"Example {name} failed"


# =============================================================================
# Import Tests (smoke tests)
# =============================================================================

class TestExampleImports:
    """Test that example utilities can be imported."""

    def test_import_alpaca_envs(self):
        """Test importing Alpaca environments."""
        from torchtrade.envs.live.alpaca.env import (
            AlpacaTorchTradingEnv,
        )
        from torchtrade.envs.live.alpaca.order_executor import (
            AlpacaOrderClass,
        )
        from torchtrade.envs.live.alpaca.observation import AlpacaObservationClass

        assert AlpacaTorchTradingEnv is not None
        assert AlpacaOrderClass is not None
        assert AlpacaObservationClass is not None

    def test_import_sampler(self):
        """Test importing the data sampler."""
        from torchtrade.envs.offline.infrastructure.sampler import MarketDataObservationSampler
        assert MarketDataObservationSampler is not None

    def test_import_utils(self):
        """Test importing utility functions."""
        from torchtrade.envs.offline.infrastructure.utils import (
            TimeFrame,
            TimeFrameUnit,
        )
        assert TimeFrame is not None
        assert TimeFrameUnit is not None
