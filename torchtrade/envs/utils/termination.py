"""Shared episode-termination rules."""


def is_bankrupt(*, current: float, initial: float, threshold: float, enabled: bool) -> bool:
    """Bankrupt when the account has fallen below `threshold` of what it started with.

    The rule for the SCALAR envs -- the live exchanges and Polymarket. The offline envs do
    their own, batched over tensors, and are not covered by this.

    What counts as "current" and "initial" is the CALLER's business and genuinely differs: the
    exchange envs mark an open position into a portfolio value, while Polymarket has no carried
    position and compares cash. Only the arithmetic is shared, so only the arithmetic lives here.

    Keyword-only: `current` and `initial` are two same-typed floats, and swapping them is
    silent and wrong.

    Strict `<`: exactly at the threshold is not yet bankrupt.
    """
    if not enabled or initial <= 0:
        return False

    return current < threshold * initial
