"""Tests for Polymarket observation class."""

import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from datetime import datetime, timezone, timedelta


class TestPolymarketObservationClass:
    """Tests for PolymarketObservationClass."""

    @pytest.fixture
    def mock_clob_client(self):
        client = MagicMock()
        client.get_midpoint.return_value = "0.72"
        client.get_order_book.return_value = MagicMock(
            bids=[MagicMock(price="0.70", size="1000")],
            asks=[MagicMock(price="0.73", size="500")],
        )
        return client

    @pytest.fixture
    def observer(self, mock_clob_client):
        patcher = patch("torchtrade.envs.live.polymarket.observation.requests.get")
        mock_get = patcher.start()

        mock_resp = MagicMock()
        end_date = (datetime.now(timezone.utc) + timedelta(days=30)).isoformat()
        mock_resp.json.return_value = {
            "id": "517310",
            "active": True,
            "closed": False,
            "volume24hr": 50000.0,
            "liquidity": 200000.0,
            "endDate": end_date,
            "outcomePrices": '["0.72", "0.28"]',
            "clobTokenIds": '["tok_yes", "tok_no"]',
        }
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        from torchtrade.envs.live.polymarket.observation import (
            PolymarketObservationClass,
        )

        obs = PolymarketObservationClass(
            yes_token_id="tok_yes",
            market_slug="test-market",
            clob_client=mock_clob_client,
        )
        yield obs
        patcher.stop()

    def test_get_observations_returns_market_state(self, observer):
        """get_observations() returns dict with 'market_state' key."""
        obs = observer.get_observations()
        assert "market_state" in obs
        state = obs["market_state"]
        assert state.shape == (5,)
        assert state.dtype == np.float32

    def test_market_state_values(self, observer):
        """market_state contains yes_price, spread, volume, liquidity, time_to_resolution."""
        obs = observer.get_observations()
        state = obs["market_state"]
        assert state[0] == pytest.approx(0.72, abs=0.01)
        assert state[1] == pytest.approx(0.03, abs=0.01)
        assert state[2] > 0
        assert state[3] > 0
        assert 0.0 < state[4] <= 1.0

    def test_get_observation_spec(self, observer):
        """get_observation_spec returns correct spec for market_state."""
        spec = observer.get_observation_spec()
        assert "market_state" in spec
        assert spec["market_state"].shape == (5,)

    def test_is_market_closed_false(self, observer):
        """is_market_closed() returns False for active market."""
        assert observer.is_market_closed() is False

    def test_is_market_closed_true(self, mock_clob_client):
        """is_market_closed() returns True when market is resolved."""
        with patch(
            "torchtrade.envs.live.polymarket.observation.requests.get"
        ) as mock_get:
            mock_resp = MagicMock()
            mock_resp.json.return_value = {
                "id": "517310",
                "active": False,
                "closed": True,
                "volume24hr": 50000.0,
                "liquidity": 200000.0,
                "endDate": "2020-01-01T00:00:00Z",
                "outcomePrices": '["1.00", "0.00"]',
                "clobTokenIds": '["tok_yes", "tok_no"]',
            }
            mock_resp.raise_for_status = MagicMock()
            mock_get.return_value = mock_resp

            from torchtrade.envs.live.polymarket.observation import (
                PolymarketObservationClass,
            )

            obs = PolymarketObservationClass(
                yes_token_id="tok_yes",
                market_slug="test-market",
                clob_client=mock_clob_client,
            )
            assert obs.is_market_closed() is True

    def test_get_yes_price(self, observer):
        """get_yes_price() returns current YES midpoint."""
        price = observer.get_yes_price()
        assert price == pytest.approx(0.72, abs=0.01)

    def test_no_token_id_resolved(self, observer):
        """no_token_id is resolved from Gamma API metadata."""
        assert observer.no_token_id == "tok_no"
