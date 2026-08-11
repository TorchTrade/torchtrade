"""
Tests for training examples.

This module tests that training examples run without errors using:
1. Mock environments for online (Alpaca) examples
2. Synthetic data for offline examples
3. Minimal training parameters for quick validation

Similar to TorchRL's sota-tests approach.
"""

import os
import shlex
import signal
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import numpy as np
import pandas as pd

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


def _check_hf_market_data_available():
    """Metadata only -- load_dataset here would pull ~1.4M rows at collection time."""
    import warnings
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    try:
        from huggingface_hub import HfApi
        HfApi().dataset_info(HF_MARKET_DATA_PATH, token=token)
        return True
    except Exception as e:
        warnings.warn(
            f"HF market data '{HF_MARKET_DATA_PATH}' check failed "
            f"(token={'set' if token else 'NOT SET'}): {e}",
            UserWarning
        )
        return False


# The example smoke tests are the ONLY thing covering the fixes that make the offline
# IQL example run at all, and they skip when the dataset is unreachable -- i.e. they
# fail open, which is exactly how the dead dataset paths stayed green for so long.
# TORCHTRADE_REQUIRE_HF=1 (set in CI) turns that skip into a real failure, and
# short-circuits the probe so CI never pays for a result it would discard.
_SKIP_HF_EXAMPLES = (
    os.environ.get("TORCHTRADE_REQUIRE_HF") != "1"
    and not _check_hf_market_data_available()
)


# =============================================================================
# Offline data plumbing (dataset_to_td + replay buffer)
# =============================================================================


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


@pytest.mark.parametrize("data_path,goes_to_hub", [
    ("Some-Org/some-transitions", True),   # org/name repo id
    ("./buf", False),                      # dot-relative on-disk buffer
    ("buf", False),                        # bare filename
    ("<abs>", False),                      # absolute path (resolved below)
], ids=["repo-id", "dot-relative", "bare-name", "absolute"])
def test_offline_buffer_routes_repo_ids_to_the_hub_and_paths_to_disk(
    tmp_path, monkeypatch, hf_dataset, iql_offline_utils, data_path, goes_to_hub
):
    """A path written as a path must never be shipped to the Hub. Probing the filesystem
    can't decide this -- hydra chdirs into its run dir before this runs -- so the routing
    is syntactic. One row per condition, so none of them can rot unnoticed."""
    import datasets
    from tensordict import TensorDict

    monkeypatch.chdir(tmp_path)
    n = 8
    TensorDict({
        "observation": torch.randn(n, 3),
        "action": torch.randint(0, 3, (n,)),
        "next": TensorDict({
            "observation": torch.randn(n, 3),
            "reward": torch.randn(n),
            "done": torch.zeros(n, dtype=torch.bool),
        }, batch_size=[n]),
    }, batch_size=[n]).save("buf")

    reached_hub = False

    def _fake_load_dataset(*a, **kw):
        nonlocal reached_hub
        reached_hub = True
        return hf_dataset

    monkeypatch.setattr(datasets, "load_dataset", _fake_load_dataset)

    if data_path == "<abs>":
        data_path = str(tmp_path / "buf")
    rb_cfg = SimpleNamespace(data_path=data_path, buffer_size=n, batch_size=4)
    iql_offline_utils.make_offline_replay_buffer(rb_cfg, env=None)

    assert reached_hub is goes_to_hub


def test_offline_iql_config_builds_a_spot_env_whose_action_head_matches(iql_offline_utils):
    """Loads the shipped config.yaml, so it also fails if a key the code reads was
    deleted. The env must be spot (flat/long) -- pairing a short level with leverage=1
    makes the env clip it to flat, giving the policy a dead action -- and the model's
    action head must be derived from the env, not hardcoded, or the two silently desync
    (a wrong head trains fine and never raises)."""
    from omegaconf import OmegaConf

    cfg = OmegaConf.load(REPO_ROOT / "examples" / "offline_rl" / "iql" / "config.yaml")
    OmegaConf.update(cfg, "logger.exp_name", "test", merge=False)

    n = 3000
    rng = np.random.default_rng(0)
    price = 100 + np.cumsum(rng.standard_normal(n) * 0.1)
    df = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=n, freq="1min"),
        "open": price, "high": price + 0.5, "low": price - 0.5,
        "close": price, "volume": rng.random(n) * 10,
    })

    raw_env = iql_offline_utils.env_maker(df, cfg)
    assert raw_env.action_levels == [0, 1], f"not spot: {raw_env.action_levels}"
    assert raw_env.leverage == 1

    # Build it the way the example does -- the encoders need the batch dim.
    train_env, eval_env = iql_offline_utils.make_environment(df, df, cfg)
    try:
        model = iql_offline_utils.make_discrete_iql_model(cfg, eval_env, torch.device("cpu"))
        obs = eval_env.reset()
        assert model[0](obs)["logits"].shape[-1] == eval_env.action_spec.n
        assert model[1](obs)["state_action_value"].shape[-1] == eval_env.action_spec.n
    finally:
        for e in (train_env, eval_env):
            if not e.is_closed:
                e.close()


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
    Run an example command (deliberately with no shell) and return its exit code.

    Args:
        command: The command to run
        timeout: Timeout in seconds

    Returns:
        Exit code (0 for success)
    """
    env = os.environ.copy()
    env["WANDB_MODE"] = "disabled"  # Disable wandb logging

    argv = shlex.split(command)
    # Use the interpreter running the tests; a stale `python` on PATH would smoke-test
    # the examples against a different torchrl than the one under test.
    if argv and argv[0] == "python":
        argv[0] = sys.executable

    # No shell: with shell=True the child of Popen is /bin/sh, so killing it on timeout
    # leaves the training run itself alive -- the actual bug in #312. start_new_session
    # puts the example in its own group so the killpg below reaches whatever it spawned
    # in turn, whether or not that thing cleans up after its own parent.
    process = subprocess.Popen(
        argv,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        cwd=str(REPO_ROOT),
        env=env,
        start_new_session=True,
    )

    try:
        stdout, _ = process.communicate(timeout=timeout)
        if process.returncode != 0:
            print(f"Command failed with exit code {process.returncode}")
            print(stdout.decode() if stdout else "")
        return process.returncode
    except BaseException:
        # BaseException, not TimeoutExpired: start_new_session took the example out of
        # pytest's process group, so a terminal Ctrl-C no longer reaches it. Catching only
        # the timeout would close that door and open this one -- and aborting a slow run by
        # hand is far more common than hitting the 300s budget.
        #
        # Not `finally`: on the success path communicate() has already reaped the child,
        # and getpgid on a reaped pid raises ProcessLookupError, which would then crash
        # every passing test.
        os.killpg(os.getpgid(process.pid), signal.SIGKILL)
        process.communicate()  # reap, or it lingers as a zombie for the session
        raise


@pytest.mark.skipif(
    _SKIP_HF_EXAMPLES,
    reason=f"HuggingFace market data '{HF_MARKET_DATA_PATH}' not accessible "
           f"(set TORCHTRADE_REQUIRE_HF=1 to fail instead of skip)"
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


def test_run_command_kills_what_the_example_spawned(tmp_path):
    """A timed-out example must not leave a training process behind (#312).

    The original bug: shell=True made Popen's child /bin/sh, so process.kill() killed the
    shell and the run survived -- burning CPU for the rest of the session under exactly
    the load that timed it out. This is the only test that fails if shell=True returns.
    """
    import time

    pidfile = tmp_path / "grandchild.pid"
    src = ("import subprocess,sys,time;"
           "p=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)']);"
           f"open({str(pidfile)!r},'w').write(str(p.pid));"
           "time.sleep(60)")

    with pytest.raises(subprocess.TimeoutExpired):
        run_command(f'python -c "{src}"', timeout=3)

    grandchild = int(pidfile.read_text())
    for _ in range(50):  # reparenting and reaping are not instant
        try:
            os.kill(grandchild, 0)
        except ProcessLookupError:
            return
        time.sleep(0.1)
    os.kill(grandchild, signal.SIGKILL)
    pytest.fail(f"grandchild {grandchild} survived the timeout")
