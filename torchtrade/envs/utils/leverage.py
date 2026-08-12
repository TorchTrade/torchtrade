"""The boundary check on the leverage a venue actually applied (#277).

Every exchange's set-leverage call was best-effort: wrapped in `except Exception`,
logged at warning, and followed by trading that assumed the request had been honoured.
`self.leverage` then stayed at the configured value whatever the venue did, which is
wrong in both directions at once -- sizing computes a target against leverage the
account does not have, and `account_state[4]` reports the config value while flat but
the venue's value once a position opens, so the element the policy conditions on flips
between two numbers with no trade in between.
"""


# The venues report "the leverage is already what you asked for" as an API *error*.
# That is a success -- the account ends up at the requested leverage either way -- and
# it is the case the blanket `except Exception: warning` existed to tolerate. Keeping
# the tolerance narrow is what lets every other failure raise.
_ALREADY_APPLIED = (
    "110043",                        # bybit: "Set leverage not modified"
    "-4046",                         # binance: "No need to change leverage"
    "leverage not modified",
    "no need to change leverage",
)


def is_already_applied(error) -> bool:
    """True when a set-leverage failure only means the venue was already there."""
    text = str(error).lower()
    return any(marker in text for marker in _ALREADY_APPLIED)


def require_leverage_applied(symbol: str, requested: float, reported) -> None:
    """Raise unless `reported` is the leverage that was asked for.

    `reported` is whatever the venue echoed back in its set-leverage response, or None
    when it echoed nothing. None is accepted: it means unconfirmed, not refused, and a
    venue that declines to state the applied leverage leaves nothing to compare against.
    A rejected *call* is a separate signal and is raised by the caller.

    Compared as floats and not `==` on the raw value because the venues disagree on the
    type: bybit sends "10", binance sends 10, ccxt sends 10.0.
    """
    if reported is None:
        return

    # A non-numeric echo is a broken adapter, not a leverage -- let the ValueError from
    # float() surface rather than comparing it and concluding "mismatch".
    if float(reported) != float(requested):
        raise ValueError(
            f"{symbol}: requested {requested}x leverage but the venue applied "
            f"{reported}x. Sizing and the account_state leverage element would both "
            f"be computed from a leverage this account does not have."
        )
