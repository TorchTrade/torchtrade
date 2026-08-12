"""The boundary check on the leverage a venue actually applied (#277)."""


# bybit alone needs a carve-out: it reports "already at this leverage" as an API error,
# and that success-shaped-as-failure is the one case the old blanket `except Exception`
# around every set-leverage call legitimately absorbed. binance, bitget and okx are
# idempotent here -- repeating the current leverage returns success.
#
# Matched on message text and NOT on bybit's 110043: that code is a substring of
# millisecond timestamps like 1762110043251, which ccxt and requests both embed in the
# error text they stringify, so a genuine refusal at an unlucky millisecond would read
# as "already applied".
_ALREADY_SET = ("leverage not modified",)


def leverage_already_set(error) -> bool:
    """True when a set-leverage failure only means the venue was already there."""
    return any(marker in str(error).lower() for marker in _ALREADY_SET)


def require_leverage_applied(symbol: str, requested: float, reported) -> None:
    """Raise unless `reported` is the leverage that was asked for.

    None is accepted deliberately: it means the venue echoed nothing for this field,
    which is unconfirmed rather than refused. A refused *call* is a separate signal,
    raised by the caller.
    """
    if reported is None:
        return

    # Coerced because the venues disagree on the type -- bybit sends "10", binance sends
    # 10, ccxt sends "25". A non-numeric echo is a broken adapter, and the bare
    # "could not convert string to float" it would otherwise raise names neither the
    # symbol nor the handshake it came from.
    try:
        applied = float(reported)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"{symbol}: venue echoed a non-numeric leverage {reported!r} for a "
            f"{requested}x request; cannot confirm the account's leverage."
        ) from error

    # NaN != anything is True, so a NaN on either side raises rather than passing --
    # the right polarity for a money path (see the `x <= 0` vs isfinite history).
    if applied != float(requested):
        raise ValueError(
            f"{symbol}: requested {requested}x leverage but the venue applied "
            f"{reported}x. Sizing and the account_state leverage element would both "
            f"be computed from a leverage this account does not have."
        )


def require_dict_response(symbol: str, requested: float, response):
    """Return `response`, or raise if it is not a shape the leverage echo can be read from.

    Skipping silently is what the check exists to prevent: these executors advertise
    `client=` injection, and a shim returning None on a 200 would erase the whole
    verification with no diagnostic.
    """
    if not isinstance(response, dict):
        raise TypeError(
            f"{symbol}: set-leverage returned {type(response).__name__}, not a dict; "
            f"cannot confirm the venue applied {requested}x."
        )
    return response
