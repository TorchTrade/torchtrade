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


def test_bitget_reads_the_leverage_of_the_margin_mode_actually_in_force():
    """bitget stores leverage per margin mode, and the body reports all of them.

    The side fields agree with the request here and only the cross field disagrees, so
    this passes if and only if the crossed body selects `crossMarginLeverage`. Every
    other fixture in the repo is isolated, which left this branch executing in no test
    at all -- the same shape of gap that let the first version of the check ship inert.
    """
    crossed = {
        "code": "00000",
        "data": {
            "symbol": "BTCUSDT", "marginCoin": "USDT",
            "longLeverage": "20", "shortLeverage": "20",
            "crossMarginLeverage": "5", "marginMode": "crossed",
        },
    }
    with pytest.raises(ValueError, match="but the venue applied"):
        _build("bitget", 20, lambda *a, **k: crossed)


# A response the venue DID return, whose leverage field is missing or renamed. Distinct
# from a non-dict response: the call looks entirely successful.
_ECHO_MISSING = {
    "binance": {"symbol": "BTCUSDT", "maxNotionalValue": "1000"},
    "bitget": {"code": "00000", "data": {"symbol": "BTCUSDT", "marginMode": "isolated"}},
    "okx": {"code": "0", "data": [{"instId": "BTC-USDT-SWAP"}]},
}


@pytest.mark.parametrize("exchange", sorted(_ECHO_MISSING))
def test_a_venue_that_reports_no_leverage_confirms_nothing(exchange):
    """Absent is not "close enough".

    Reading a key that does not exist, finding None and returning happy is exactly how
    the first version of this check shipped inert on bitget. A renamed or dropped field
    has to fail loudly, or the check silently stops checking and the suite stays green.
    """
    with pytest.raises(ValueError, match="did not report"):
        _build(exchange, 20, lambda *a, **k: _ECHO_MISSING[exchange])


def test_okx_confirms_nothing_when_it_returns_no_entries():
    """An empty data list is a failure to confirm, not a pass."""
    with pytest.raises(ValueError, match="confirmed no leverage"):
        _build("okx", 20, lambda *a, **k: {"code": "0", "data": []})


def test_bitget_confirms_nothing_on_the_unified_account_route():
    """The unified route answers `"data": "success"` -- a string, carrying no leverage."""
    with pytest.raises(ValueError, match="confirmed no leverage"):
        _build("bitget", 20, lambda *a, **k: {"code": "00000", "data": "success"})


@pytest.mark.parametrize("exchange", ["bitget", "bybit"])
def test_the_margin_mode_is_applied_before_the_leverage_it_can_overwrite(exchange):
    """Verifying leverage first confirms a number that stops being true one call later.

    bitget stores leverage per margin mode, so a leverage verified under isolated says
    nothing about the account once it switches to crossed. bybit is worse: its
    switch_margin_mode carries buyLeverage and sellLeverage, and it is tolerated with a
    warning, so it can silently undo the verified set.

    Order is asserted directly because neither effect is observable from construction --
    both leave a *successful* build holding a leverage nobody checked.
    """
    calls = []

    def record(name, result):
        def call(*a, **k):
            calls.append(name)
            return result
        return call

    if exchange == "bitget":
        from torchtrade.envs.live.bitget.order_executor import BitgetFuturesOrderClass
        BitgetFuturesOrderClass(
            symbol="BTCUSDT", trade_mode="quantity", demo=True, leverage=20,
            client=SimpleNamespace(
                set_leverage=record("leverage", {"code": "00000", "data": {
                    "longLeverage": "20", "shortLeverage": "20",
                    "crossMarginLeverage": "20", "marginMode": "isolated"}}),
                set_margin_mode=record("margin_mode", {}),
                set_position_mode=_boom, load_markets=_boom, markets={},
                fetch_positions=_boom,
            ),
        )
    else:
        from torchtrade.envs.live.bybit.order_executor import (
            BybitFuturesOrderClass, MarginMode, PositionMode,
        )
        BybitFuturesOrderClass(
            symbol="BTCUSDT", trade_mode="quantity", demo=True, leverage=20,
            margin_mode=MarginMode.ISOLATED, position_mode=PositionMode.ONE_WAY,
            api_key="k", api_secret="s",
            client=SimpleNamespace(
                set_leverage=record("leverage", {"retCode": 0}),
                switch_margin_mode=record("margin_mode", {"retCode": 0}),
                switch_position_mode=_boom, get_instruments_info=_boom, get_positions=_boom,
            ),
        )

    assert calls.index("margin_mode") < calls.index("leverage"), (
        f"{exchange} verified leverage before setting the margin mode that can change "
        f"it; call order was {calls}"
    )


@pytest.mark.parametrize("position_mode,margin_mode,expected_sides", [
    ("LONG_SHORT", "ISOLATED", ["long", "short"]),
    ("LONG_SHORT", "CROSS", [None]),
    ("NET", "ISOLATED", [None]),
], ids=["isolated-long-short", "cross-long-short", "net-isolated"])
def test_okx_sets_leverage_per_side_only_where_okx_requires_it(
    position_mode, margin_mode, expected_sides
):
    """OKX rejects set-leverage without posSide under isolated + long_short (#363).

    Isolated leverage is stored per side there, so it takes one call each. Omitting
    posSide had the venue answer code 1 "Parameter posSide error", which #277's
    swallowed-rejection era turned into sizing against leverage the account never had.

    The other two rows are the ones that make this a real test rather than a shape
    check: sending posSide where OKX does not expect it is its own rejection, so the
    per-side path must NOT widen to cross or net.
    """
    from torchtrade.envs.live.okx.order_executor import (
        OKXFuturesOrderClass, MarginMode, PositionMode,
    )
    seen = []

    def record(**kwargs):
        pos_side = kwargs.get("posSide")
        seen.append(pos_side)
        # Echoes the side back, as OKX documents -- the executor checks it.
        entry = {"lever": "20", **({} if pos_side is None else {"posSide": pos_side})}
        return {"code": "0", "data": [entry]}

    OKXFuturesOrderClass(
        symbol="BTC-USDT-SWAP", trade_mode="quantity", demo=True, leverage=20,
        margin_mode=getattr(MarginMode, margin_mode),
        position_mode=getattr(PositionMode, position_mode),
        api_key="k", api_secret="s", passphrase="p",
        client=SimpleNamespace(),
        account_client=SimpleNamespace(set_leverage=record, set_position_mode=_boom,
                                       get_positions=_boom),
        public_client=SimpleNamespace(get_instruments=_boom),
    )
    # Order-insensitive: the two calls are independent and nothing in OKX's contract
    # makes long precede short, so pinning the sequence would break a harmless refactor.
    assert sorted(seen, key=str) == sorted(expected_sides, key=str), (
        f"{margin_mode}/{position_mode} should set leverage for {expected_sides}, got {seen}"
    )


def test_okx_refuses_to_construct_when_only_one_side_took_the_leverage():
    """Two calls create a partial-failure state one call could not (#363).

    Long accepted at 20x and short refused leaves the account genuinely half-configured,
    while the env sizes BOTH sides from a single config.leverage. Swallowing the second
    leg's rejection passed all 31 tests in this file, so the behaviour was correct and
    unpinned.

    What this proves is that the Python object refuses to exist. It cannot prove the
    venue rolls back the long leg that already applied -- that is outside the executor's
    control, and an operator hitting this has one side set and must fix it by hand.
    """
    from torchtrade.envs.live.okx.order_executor import (
        OKXFuturesOrderClass, MarginMode, PositionMode,
    )

    def one_sided(**kwargs):
        if kwargs.get("posSide") == "short":
            return {"code": "51004", "data": [{"sMsg": "leverage exceeds risk limit"}]}
        return {"code": "0", "data": [{"lever": "20", "posSide": kwargs.get("posSide")}]}

    with pytest.raises(ValueError, match="refused"):
        OKXFuturesOrderClass(
            symbol="BTC-USDT-SWAP", trade_mode="quantity", demo=True, leverage=20,
            margin_mode=MarginMode.ISOLATED, position_mode=PositionMode.LONG_SHORT,
            api_key="k", api_secret="s", passphrase="p",
            client=SimpleNamespace(),
            account_client=SimpleNamespace(set_leverage=one_sided,
                                           set_position_mode=_boom, get_positions=_boom),
            public_client=SimpleNamespace(get_instruments=_boom),
        )


def test_okx_refuses_a_confirmation_for_the_wrong_side():
    """Echoing `lever` is not confirming the leg that was asked about (#363).

    A response carrying posSide="long" for the SHORT call passed verification, because
    only `lever` was ever compared -- so the short side went unconfirmed while the check
    reported success. Every fake in this file echoes the CORRECT side, which is why
    disabling the comparison entirely left all 32 tests green.
    """
    from torchtrade.envs.live.okx.order_executor import (
        OKXFuturesOrderClass, MarginMode, PositionMode,
    )

    def always_long(**kwargs):
        return {"code": "0", "data": [{"lever": "20", "posSide": "long"}]}

    with pytest.raises(ValueError, match="posSide"):
        OKXFuturesOrderClass(
            symbol="BTC-USDT-SWAP", trade_mode="quantity", demo=True, leverage=20,
            margin_mode=MarginMode.ISOLATED, position_mode=PositionMode.LONG_SHORT,
            api_key="k", api_secret="s", passphrase="p",
            client=SimpleNamespace(),
            account_client=SimpleNamespace(set_leverage=always_long,
                                           set_position_mode=_boom, get_positions=_boom),
            public_client=SimpleNamespace(get_instruments=_boom),
        )
