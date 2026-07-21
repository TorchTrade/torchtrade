import pytest
from torchtrade.llm.train.losses import resolve_loss, validate_num_generations


class _StubLoss:
    def __init__(self, actor, **kw):
        self.actor = actor
        self.kw = kw


def test_grpo_name_resolves_and_forwards_kwargs(monkeypatch):
    import torchrl.objectives.llm as m
    monkeypatch.setattr(m, "GRPOLoss", _StubLoss)  # resolve_loss imports GRPOLoss lazily at call time
    loss = resolve_loss("grpo", actor_network="ACTOR", loss_kwargs={"clip_epsilon": 0.2})
    assert isinstance(loss, _StubLoss)
    assert loss.actor == "ACTOR" and loss.kw == {"clip_epsilon": 0.2}


def test_sao_name_resolves_and_forwards_kwargs(monkeypatch):
    import torchtrade.losses.sao_loss as m
    monkeypatch.setattr(m, "SAOLoss", _StubLoss)  # resolve_loss imports SAOLoss lazily at call time
    loss = resolve_loss("sao", actor_network="ACTOR",
                        loss_kwargs={"epsilon_low": 0.3, "epsilon_high": 5.0})
    assert isinstance(loss, _StubLoss)
    assert loss.actor == "ACTOR" and loss.kw == {"epsilon_low": 0.3, "epsilon_high": 5.0}


def test_callable_factory_is_used_and_receives_kwargs():
    """resolve_loss calls the factory with actor_network (identity passthrough) AND threads
    loss_kwargs into it (needed for SAOLoss-style factories) — pinned in one test."""
    sentinel = object()

    def factory(actor, **kw):
        assert actor == "ACTOR"
        return sentinel if not kw else _StubLoss(actor, **kw)

    assert resolve_loss(factory, actor_network="ACTOR") is sentinel
    loss = resolve_loss(factory, actor_network="ACTOR", loss_kwargs={"epsilon_high": 5.0})
    assert isinstance(loss, _StubLoss) and loss.kw == {"epsilon_high": 5.0}


def test_unknown_name_raises():
    with pytest.raises(ValueError, match="unknown loss"):
        resolve_loss("does-not-exist", actor_network="ACTOR")


@pytest.mark.parametrize("n,ok", [(1, False), (2, True), (8, True)])
def test_validate_num_generations(n, ok):
    if ok:
        validate_num_generations(n)
    else:
        with pytest.raises(ValueError, match="num_generations"):
            validate_num_generations(n)
