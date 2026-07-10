from torchtrade.train.dataset import build_bar_dataset


class _FakeActor:
    def _resolve_system_prompt(self): return "SYS"
    def _construct_user_prompt(self, td): return f"USER for {td['bar_index']}"


class _FakeEnv:
    """Yields a per-bar 'observation' tensordict-like dict with bar_index."""
    def obs_at(self, i):        # implementer names this to match Task 1's deterministic access
        return {"bar_index": i}


def test_builds_one_row_per_bar():
    rows = build_bar_dataset(_FakeActor(), _FakeEnv(), n_bars=3)
    assert len(rows) == 3
    assert rows[0] == {"bar_index": 0, "prompt": "USER for 0", "system_prompt": "SYS"}
    assert [r["bar_index"] for r in rows] == [0, 1, 2]
