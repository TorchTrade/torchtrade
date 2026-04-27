"""Tests for Polymarket observation class."""

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from torchtrade.envs.live.polymarket.observation import PolymarketObservationClass

ACTIVE_MARKET_PAYLOAD = {
    "id": "517310",
    "active": True,
    "closed": False,
    "volume24hr": 50000.0,
    "liquidity": 200000.0,
    "outcomePrices": '["0.72", "0.28"]',
    "clobTokenIds": '["tok_yes", "tok_no"]',
}


def _payload(**overrides) -> dict:
    payload = dict(ACTIVE_MARKET_PAYLOAD)
    payload["endDate"] = (
        datetime.now(timezone.utc) + timedelta(days=30)
    ).isoformat()
    payload.update(overrides)
    return payload


def _mock_response(payload: dict) -> MagicMock:
    resp = MagicMock()
    resp.json.return_value = payload
    resp.raise_for_status = MagicMock()
    return resp


@pytest.fixture
def mock_clob_client():
    client = MagicMock()
    client.get_midpoint.return_value = "0.72"
    client.get_order_book.return_value = MagicMock(
        bids=[MagicMock(price="0.70", size="1000")],
        asks=[MagicMock(price="0.73", size="500")],
    )
    return client


@pytest.fixture
def observer(mock_clob_client):
    patcher = patch("torchtrade.envs.live.polymarket.observation.requests.get")
    mock_get = patcher.start()
    mock_get.return_value = _mock_response(_payload())
    obs = PolymarketObservationClass(
        yes_token_id="tok_yes",
        market_slug="test-market",
        clob_client=mock_clob_client,
    )
    yield obs
    patcher.stop()


class TestPolymarketObservationClass:
    """Tests for PolymarketObservationClass."""

    def test_get_observations_shape_and_dtype(self, observer):
        state = observer.get_observations()["market_state"]
        assert state.shape == (5,)
        assert state.dtype == np.float32

    def test_market_state_values(self, observer):
        """[yes_price, spread, volume24h, liquidity, time_to_resolution]."""
        state = observer.get_observations()["market_state"]
        assert state[0] == pytest.approx(0.72, abs=0.01)
        assert state[1] == pytest.approx(0.03, abs=0.01)
        assert state[2] > 0
        assert state[3] > 0
        assert 0.0 < state[4] <= 1.0

    def test_no_token_id_resolved(self, observer):
        assert observer.no_token_id == "tok_no"

    @pytest.mark.parametrize("closed", [False, True], ids=["open", "closed"])
    def test_is_market_closed(self, mock_clob_client, closed):
        with patch(
            "torchtrade.envs.live.polymarket.observation.requests.get"
        ) as mock_get:
            mock_get.return_value = _mock_response(
                _payload(closed=closed, active=not closed)
            )
            obs = PolymarketObservationClass(
                yes_token_id="tok_yes",
                market_slug="test-market",
                clob_client=mock_clob_client,
            )
            assert obs.is_market_closed() is closed

    @pytest.mark.parametrize(
        "kwargs,expected_param",
        [
            ({"market_slug": "the-market"}, "slug"),
            ({"condition_id": "0xabc"}, "condition_id"),
            ({}, "clob_token_ids"),  # falls back to yes_token_id
        ],
        ids=["by-slug", "by-condition", "by-token-only"],
    )
    def test_metadata_query_params(self, mock_clob_client, kwargs, expected_param):
        with patch(
            "torchtrade.envs.live.polymarket.observation.requests.get"
        ) as mock_get:
            mock_get.return_value = _mock_response(_payload())
            PolymarketObservationClass(
                yes_token_id="tok_yes",
                clob_client=mock_clob_client,
                **kwargs,
            )
        sent_params = mock_get.call_args.kwargs["params"]
        assert expected_param in sent_params

    def test_get_yes_price_uses_clob_midpoint(self, observer):
        assert observer.get_yes_price() == pytest.approx(0.72, abs=0.01)

    def test_get_yes_price_falls_back_to_gamma_when_clob_fails(self, observer):
        observer.clob_client.get_midpoint = MagicMock(side_effect=Exception("clob down"))
        # Gamma metadata says outcomePrices=["0.72","0.28"] from the fixture.
        assert observer.get_yes_price() == pytest.approx(0.72, abs=0.01)

    def test_get_yes_price_returns_neutral_when_no_clob_and_metadata_missing(
        self, mock_clob_client
    ):
        with patch(
            "torchtrade.envs.live.polymarket.observation.requests.get"
        ) as mock_get:
            payload = _payload()
            payload["outcomePrices"] = "not-json"
            mock_get.return_value = _mock_response(payload)
            obs = PolymarketObservationClass(
                yes_token_id="tok_yes", market_slug="x", clob_client=None
            )
        assert obs.get_yes_price() == 0.5

    @pytest.mark.parametrize(
        "end_date,expected_range",
        [
            ("", (1.0, 1.0)),
            ("2020-01-01T00:00:00Z", (0.0, 0.0)),
            ("garbage", (1.0, 1.0)),
        ],
        ids=["missing", "past", "malformed"],
    )
    def test_time_to_resolution_edge_cases(
        self, mock_clob_client, end_date, expected_range
    ):
        with patch(
            "torchtrade.envs.live.polymarket.observation.requests.get"
        ) as mock_get:
            mock_get.return_value = _mock_response(_payload(endDate=end_date))
            obs = PolymarketObservationClass(
                yes_token_id="tok_yes",
                market_slug="x",
                clob_client=mock_clob_client,
            )
            ttr = obs.get_observations()["market_state"][4]
        lo, hi = expected_range
        assert lo <= float(ttr) <= hi

    def test_metadata_failure_returns_empty_dict(self, mock_clob_client):
        """An HTTP failure during init returns an empty metadata dict; no_token_id stays empty."""
        with patch(
            "torchtrade.envs.live.polymarket.observation.requests.get",
            side_effect=Exception("503"),
        ):
            obs = PolymarketObservationClass(
                yes_token_id="tok_yes", market_slug="x", clob_client=mock_clob_client
            )
        assert obs.no_token_id == ""
        assert obs.is_market_closed() is False

    def test_get_observation_spec_matches_env(self, observer):
        spec = observer.get_observation_spec()
        assert "market_state" in spec
        assert spec["market_state"].shape == (5,)
