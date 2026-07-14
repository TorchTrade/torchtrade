"""Shared episode-termination rules."""


def is_bankrupt(current: float, initial: float, threshold: float, enabled: bool) -> bool:
    """Bankrupt when the account has fallen below `threshold` of what it started with.

    The one bankruptcy rule. What counts as "current" and "initial" is the CALLER's business
    and genuinely differs -- the exchange envs mark an open position into a portfolio value,
    while Polymarket has no carried position and compares cash. Only the arithmetic is shared,
    so only the arithmetic lives here.

    Strict `<`: exactly at the threshold is not yet bankrupt.
    """
    if not enabled or initial <= 0:
        return False

    return current < threshold * initial
