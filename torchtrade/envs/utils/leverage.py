"""The boundary check on the leverage a venue actually applied (#277)."""


# Venues can report "already at this leverage" as an API *error* -- the one case the old
# blanket `except Exception` around every set-leverage call legitimately absorbed.
# Confirmed on bybit; applied at all four call sites because the cost of guessing wrong
# is refusing to construct on every rerun against an already-configured account.
# Matched on the message and NOT on bybit's code 110043: that digit string also occurs
# inside millisecond timestamps such as 1762110043251, which ccxt and requests both embed
# in the error text they stringify, so a real refusal would have read as "already set".
_ALREADY_SET = "leverage not modified"


def leverage_already_set(error) -> bool:
    """True when a set-leverage failure only means the venue was already there."""
    return _ALREADY_SET in str(error).lower()


def require_dict_response(symbol: str, requested: float, response) -> None:
    """Raise unless the leverage echo can be read off `response` at all.

    Skipping silently is the failure this check exists to prevent: these executors
    accept an injected `client`, and a shim returning None on a 200 would erase the
    verification with no diagnostic.
    """
    if not isinstance(response, dict):
        raise TypeError(
            f"{symbol}: set-leverage returned {type(response).__name__}, not a dict; "
            f"cannot confirm the venue applied {requested}x."
        )


def require_leverage_applied(symbol: str, requested: float, container, key: str) -> None:
    """Raise unless `container[key]` is the leverage that was asked for.

    An ABSENT key raises rather than passing. Treating "the venue said nothing" as
    "close enough" is precisely how the first version of this check shipped inert: it
    read a bitget key that does not exist, found None, and returned happy on every
    account. A renamed or dropped field must fail loudly, not silently stop verifying.
    """
    require_dict_response(symbol, requested, container)
    if key not in container or container[key] is None:
        raise ValueError(
            f"{symbol}: venue did not report {key!r} in its set-leverage response, so "
            f"there is nothing to confirm {requested}x against. Refusing to size "
            f"positions from an unconfirmed leverage."
        )

    reported = container[key]
    # Coerced because the venues disagree on the type: binance sends 10, ccxt sends "25".
    try:
        applied = float(reported)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"{symbol}: venue echoed a non-numeric {key} {reported!r} for a "
            f"{requested}x request; cannot confirm the account\'s leverage."
        ) from error

    # NaN != anything is True, so a NaN on either side raises rather than passing.
    if applied != float(requested):
        raise ValueError(
            f"{symbol}: requested {requested}x leverage but the venue applied "
            f"{reported}x. Sizing and the account_state leverage element would both "
            f"be computed from a leverage this account does not have."
        )
