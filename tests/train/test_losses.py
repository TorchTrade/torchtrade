import pytest
from torchtrade.train.losses import resolve_loss, validate_num_generations


class _StubLoss:
    def __init__(self, actor, **kw):
        self.actor = actor
        self.kw = kw


def test_registry_name_resolves_and_forwards_kwargs(monkeypatch):
    import torchtrade.train.losses as m
    monkeypatch.setitem(m._LOSS_REGISTRY, "stub", lambda: _StubLoss)
    loss = resolve_loss("stub", actor_network="ACTOR", loss_kwargs={"clip_epsilon": 0.2})
    assert isinstance(loss, _StubLoss)
    assert loss.actor == "ACTOR" and loss.kw == {"clip_epsilon": 0.2}


def test_callable_factory_is_used():
    sentinel = object()
    loss = resolve_loss(lambda actor: sentinel, actor_network="ACTOR")
    assert loss is sentinel


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
