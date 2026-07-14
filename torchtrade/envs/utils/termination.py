"""Shared episode-termination rules."""


def is_bankrupt(*, current: float, initial: float, threshold: float, enabled: bool) -> bool:
    """Bankrupt when the account has fallen below `threshold` of what it started with.

    Keyword-only: `current` and `initial` are two same-typed floats, and swapping them is
    silent and wrong.

    Strict `<`: exactly at the threshold is not yet bankrupt.
    """
    return enabled and current < threshold * initial
