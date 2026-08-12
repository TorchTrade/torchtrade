"""The venue must confirm the leverage before the env sizes anything against it (#277).

Every exchange used to wrap its set-leverage call in `except Exception: warning`, so a
rejection left `self.leverage` at the configured value and the env went on sizing
positions against leverage the account did not have -- and reporting that value as
`account_state[4]` while flat, while reporting the venue's real one once a position
opened. Same element, two numbers, no trade in between.

These build REAL order executors, because the defect lived in each executor's
`_setup_futures_account`; a fake trader would prove nothing about it.
"""
import pytest
from types import SimpleNamespace


def _boom(*a, **k):
    raise ConnectionError("unused in these tests")


def _build(exchange, leverage, set_leverage):
    """Construct a real order executor whose leverage call is `set_leverage`.

    Every other venue call raises: each is already wrapped in its own tolerant handler,
    so construction reaching the end proves the leverage path alone decided the outcome.
    """
    if exchange == "binance":
        from torchtrade.envs.live.binance.order_executor import BinanceFuturesOrderClass
        return BinanceFuturesOrderClass(
            symbol="BTCUSDT", trade_mode="quantity", demo=True, leverage=leverage,
            client=SimpleNamespace(
                futures_change_leverage=set_leverage, futures_change_margin_type=_boom,
                futures_exchange_info=_boom, futures_position_information=_boom,
            ),
        )
    if exchange == "bitget":
        from torchtrade.envs.live.bitget.order_executor import BitgetFuturesOrderClass
        return BitgetFuturesOrderClass(
            symbol="BTCUSDT", trade_mode="quantity", demo=True, leverage=leverage,
            client=SimpleNamespace(
                set_leverage=set_leverage, set_position_mode=_boom, set_margin_mode=_boom,
                load_markets=_boom, markets={}, fetch_positions=_boom,
            ),
        )
    if exchange == "bybit":
        from torchtrade.envs.live.bybit.order_executor import (
            BybitFuturesOrderClass, MarginMode, PositionMode,
        )
        return BybitFuturesOrderClass(
            symbol="BTCUSDT", trade_mode="quantity", demo=True, leverage=leverage,
            margin_mode=MarginMode.ISOLATED, position_mode=PositionMode.ONE_WAY,
            api_key="k", api_secret="s",
            client=SimpleNamespace(
                set_leverage=set_leverage, switch_position_mode=_boom,
                switch_margin_mode=_boom, get_instruments_info=_boom, get_positions=_boom,
            ),
        )
    if exchange == "okx":
        from torchtrade.envs.live.okx.order_executor import (
            OKXFuturesOrderClass, MarginMode, PositionMode,
        )
        return OKXFuturesOrderClass(
            symbol="BTC-USDT-SWAP", trade_mode="quantity", demo=True, leverage=leverage,
            margin_mode=MarginMode.ISOLATED, position_mode=PositionMode.NET,
            api_key="k", api_secret="s", passphrase="p",
            client=SimpleNamespace(),
            account_client=SimpleNamespace(
                set_leverage=set_leverage, set_position_mode=_boom, get_positions=_boom,
            ),
            public_client=SimpleNamespace(get_instruments=_boom),
        )
    raise AssertionError(f"unhandled exchange {exchange}")


EXCHANGES = ["binance", "bitget", "bybit", "okx"]

# How each venue states the leverage it actually applied. bybit is absent because its
# set-leverage response carries no leverage field -- it can only be checked for refusal,
# which is what makes the refusal path below the one that has to hold on all four.
_ECHO = {
    "binance": lambda applied: {"leverage": applied},
    # ccxt returns bitget's raw body: the applied leverage lives per side under `data`,
    # and there is no top-level `leverage` key. Shaping this to the code instead of the
    # venue is what made an inert check look verified.
    "bitget": lambda applied: {
        "code": "00000", "msg": "success", "requestTime": 1700864711517,
        "data": {
            "symbol": "BTCUSDT", "marginCoin": "USDT",
            "longLeverage": str(applied), "shortLeverage": str(applied),
            "crossMarginLeverage": str(applied), "marginMode": "isolated",
        },
    },
    "okx": lambda applied: {"code": "0", "data": [{"lever": str(applied)}]},
}


@pytest.mark.parametrize("exchange", EXCHANGES)
def test_a_refused_leverage_stops_construction(exchange):
    """The headline defect: the refusal was logged at warning and trading continued."""
    def refuse(*a, **k):
        raise ConnectionError("venue refused the leverage")

    with pytest.raises(ConnectionError):
        _build(exchange, 20, refuse)


@pytest.mark.parametrize("exchange", EXCHANGES)
def test_leverage_the_venue_already_had_is_not_a_refusal(exchange):
    """Every venue reports "already at this value" as an API error.

    Without this carve-out the fix above would refuse to construct on every rerun
    against an account already configured -- which is the reason the original blanket
    `except Exception` was there, and why removing it needs this case to stay green.
    """
    def already(*a, **k):
        raise RuntimeError("ErrCode: 110043 leverage not modified")

    assert _build(exchange, 20, already).leverage == 20


@pytest.mark.parametrize("exchange", sorted(_ECHO))
def test_a_venue_that_applies_a_different_leverage_is_refused(exchange):
    """Accepting the call is not applying the request.

    Binance caps leverage per notional bracket, so "accepted" can still mean the account
    sits at a lower leverage than the one every position size is computed from.
    """
    echo = _ECHO[exchange]
    with pytest.raises(ValueError, match="but the venue applied"):
        _build(exchange, 20, lambda *a, **k: echo(5))


# The two venues that report a refusal as a status code in an otherwise-successful
# response rather than by raising. Both adapters check codes this way everywhere else.
_REJECTION_CODE = {
    "bybit": {"retCode": 110045, "retMsg": "risk limit exceeded"},
    "okx": {"code": "51004", "msg": "leverage exceeds the risk limit"},
}


@pytest.mark.parametrize("exchange", sorted(_REJECTION_CODE))
def test_a_refusal_reported_as_a_status_code_stops_construction(exchange):
    """A refusal need not arrive as an exception; these venues return one in the body."""
    rejection = _REJECTION_CODE[exchange]
    with pytest.raises(ValueError, match="refused"):
        _build(exchange, 20, lambda *a, **k: rejection)


@pytest.mark.parametrize("exchange", sorted(_ECHO))
def test_a_venue_that_confirms_the_request_constructs(exchange):
    """The other half of the check: a matching echo must not be read as a mismatch."""
    echo = _ECHO[exchange]
    assert _build(exchange, 20, lambda *a, **k: echo(20)).leverage == 20


@pytest.mark.parametrize("exchange", EXCHANGES)
def test_a_response_the_echo_cannot_be_read_from_stops_construction(exchange):
    """These executors advertise `client=` injection, so the shape is not guaranteed.

    A shim returning None on a 200 used to erase the whole verification with no
    diagnostic -- the check silently skipped and construction reported success.
    """
    with pytest.raises(TypeError, match="not a dict"):
        _build(exchange, 20, lambda *a, **k: None)
