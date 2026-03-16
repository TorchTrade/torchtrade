# PolyTimeBarEnv Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement a Polymarket live trading environment (`PolyTimeBarEnv`) that trades YES/NO shares on a single prediction market, stepping on regular time bars, with optional supplementary data sources.

**Architecture:** Observer/Trader dependency injection following the Alpaca pattern. `PolymarketObservationClass` fetches market state from Gamma API + CLOB. `PolymarketOrderExecutor` wraps `py-clob-client` for trade execution. `PolyTimeBarEnv` orchestrates the step loop. Supplementary observers (e.g., Binance OHLCV) can be composed in at construction time.

**Tech Stack:** `py-clob-client` (Polymarket CLOB API), `requests` (Gamma API), TorchRL (`EnvBase`, TensorDict), existing TorchTrade base classes (`TorchTradeLiveEnv`, `PositionState`, `HistoryTracker`)

**Spec:** `docs/superpowers/specs/2026-03-16-polymarket-integration-design.md`

**Already implemented:** `market_scanner.py` + tests (Task 1 complete, 19 tests passing)

---

## Chunk 1: Test Mocks + Order Executor

### Task 1: Test Mocks for Polymarket

**Files:**
- Create: `tests/envs/polymarket/mocks.py`

These mocks simulate the Polymarket CLOB client and Gamma API responses. They are used by all subsequent test files.

- [ ] **Step 1: Write mock infrastructure**

```python
# tests/envs/polymarket/mocks.py
"""Mock infrastructure for Polymarket environment tests."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from unittest.mock import MagicMock

import numpy as np


# --- Mock CLOB Client ---

@dataclass
class MockPosition:
    """Simulated Polymarket share position."""
    token_id: str = ""
    size: float = 0.0
    avg_cost: float = 0.0
    cur_cost: float = 0.0


@dataclass
class MockOrderBook:
    """Simulated order book."""
    bids: list = field(default_factory=lambda: [{"price": "0.70", "size": "1000"}])
    asks: list = field(default_factory=lambda: [{"price": "0.73", "size": "500"}])


class MockClobClient:
    """Simulates py-clob-client's ClobClient for testing.

    Tracks balance and positions locally. Supports buy/sell
    with price-aware fill simulation.
    """

    def __init__(self, initial_balance: float = 10_000.0, yes_price: float = 0.72):
        self.balance = initial_balance
        self.yes_price = yes_price
        self.positions: Dict[str, MockPosition] = {}
        self.orders: list = []

    def create_or_derive_api_creds(self):
        return {"apiKey": "test-key", "secret": "test-secret", "passphrase": "test-pass"}

    def set_api_creds(self, creds):
        pass

    def get_balance_allowance(self, **kwargs):
        # Balance in USDC atomic units (6 decimals)
        return {"balance": str(int(self.balance * 1e6))}

    def create_market_order(self, order_args):
        return MagicMock(order=order_args)

    def post_order(self, signed_order, order_type=None):
        args = signed_order.order
        token_id = args.token_id
        amount = args.amount
        side = args.side

        if side == "BUY":
            cost = amount
            if cost > self.balance:
                raise Exception("Insufficient balance")
            shares = cost / self.yes_price
            self.balance -= cost
            pos = self.positions.get(token_id, MockPosition(token_id=token_id))
            pos.size += shares
            pos.avg_cost = self.yes_price
            pos.cur_cost = self.yes_price
            self.positions[token_id] = pos
        else:  # SELL
            pos = self.positions.get(token_id)
            if pos is None or pos.size < amount:
                raise Exception("Insufficient shares")
            revenue = amount * self.yes_price
            pos.size -= amount
            self.balance += revenue
            if pos.size <= 0:
                del self.positions[token_id]

        return {"success": True, "orderID": f"order-{len(self.orders)}"}

    def get_order_book(self, token_id):
        return MockOrderBook()

    def get_midpoint(self, token_id):
        return str(self.yes_price)

    def cancel_all(self):
        self.orders.clear()
        return {"success": True}

    def get_positions(self, **kwargs):
        return [
            {"asset": pos.token_id, "size": str(pos.size), "avgCost": str(pos.avg_cost)}
            for pos in self.positions.values()
        ]


# --- Mock Observer ---

class MockPolymarketObserver:
    """Simulates PolymarketObservationClass for testing."""

    def __init__(self, yes_price: float = 0.72, spread: float = 0.02,
                 volume_24h: float = 50_000.0, liquidity: float = 200_000.0,
                 time_to_resolution: float = 0.8, market_closed: bool = False):
        self.yes_price = yes_price
        self.spread = spread
        self.volume_24h = volume_24h
        self.liquidity = liquidity
        self.time_to_resolution = time_to_resolution
        self.market_closed = market_closed
        self.yes_token_id = "tok_yes_1"
        self.no_token_id = "tok_no_1"

    def get_observations(self) -> dict:
        return {
            "market_state": np.array([
                self.yes_price,
                self.spread,
                self.volume_24h,
                self.liquidity,
                self.time_to_resolution,
            ], dtype=np.float32),
        }

    def get_observation_spec(self) -> dict:
        from torchrl.data import Bounded
        import torch
        return {
            "market_state": Bounded(
                low=0.0, high=float("inf"), shape=(5,), dtype=torch.float32
            ),
        }

    def is_market_closed(self) -> bool:
        return self.market_closed

    def get_yes_price(self) -> float:
        return self.yes_price


# --- Mock Trader ---

class MockPolymarketTrader:
    """Simulates PolymarketOrderExecutor for testing."""

    def __init__(self, initial_balance: float = 10_000.0, yes_price: float = 0.72):
        self.client = MockClobClient(initial_balance=initial_balance, yes_price=yes_price)
        self.yes_token_id = "tok_yes_1"
        self.no_token_id = "tok_no_1"
        self._dry_run = False

    def get_balance(self) -> float:
        raw = float(self.client.get_balance_allowance()["balance"])
        return raw / 1e6

    def buy(self, token_id: str, amount_usdc: float) -> dict:
        if self._dry_run:
            return {"success": True, "dry_run": True}
        try:
            args = MagicMock(token_id=token_id, amount=amount_usdc, side="BUY")
            signed = MagicMock(order=args)
            result = self.client.post_order(signed)
            return {"success": True, "order": result}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def sell(self, token_id: str, amount_shares: float) -> dict:
        if self._dry_run:
            return {"success": True, "dry_run": True}
        try:
            args = MagicMock(token_id=token_id, amount=amount_shares, side="SELL")
            signed = MagicMock(order=args)
            result = self.client.post_order(signed)
            return {"success": True, "order": result}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def get_positions(self) -> list:
        return self.client.get_positions()

    def cancel_all(self) -> bool:
        self.client.cancel_all()
        return True

    def get_yes_price(self) -> float:
        return self.client.yes_price

    def close_position(self, token_id: str) -> dict:
        pos = self.client.positions.get(token_id)
        if pos is None or pos.size <= 0:
            return {"success": True, "already_flat": True}
        return self.sell(token_id, pos.size)
```

- [ ] **Step 2: Verify mocks import correctly**

Run: `uv run python -c "from tests.envs.polymarket.mocks import MockClobClient, MockPolymarketObserver, MockPolymarketTrader; print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add tests/envs/polymarket/mocks.py
git commit -m "feat(polymarket): add mock infrastructure for Polymarket env tests"
```

---

### Task 2: Order Executor

**Files:**
- Create: `torchtrade/envs/live/polymarket/order_executor.py`
- Test: `tests/envs/polymarket/test_order_executor.py`

Wraps `py-clob-client` for Polymarket CLOB trading. Supports buy/sell of YES/NO shares, balance queries, position queries, and dry-run mode.

- [ ] **Step 1: Write the failing test**

```python
# tests/envs/polymarket/test_order_executor.py
"""Tests for Polymarket order executor."""

import pytest
from unittest.mock import MagicMock, patch

from tests.envs.polymarket.mocks import MockClobClient


class TestPolymarketOrderExecutor:
    """Tests for PolymarketOrderExecutor."""

    @pytest.fixture
    def mock_clob(self):
        return MockClobClient(initial_balance=10_000.0, yes_price=0.72)

    @pytest.fixture
    def executor(self, mock_clob):
        with patch(
            "torchtrade.envs.live.polymarket.order_executor.ClobClient",
            return_value=mock_clob,
        ):
            from torchtrade.envs.live.polymarket.order_executor import (
                PolymarketOrderExecutor,
            )

            exe = PolymarketOrderExecutor(private_key="0xtest", chain_id=137)
            exe.client = mock_clob
            return exe

    def test_get_balance(self, executor):
        """get_balance returns USDC balance as float dollars."""
        balance = executor.get_balance()
        assert isinstance(balance, float)
        assert balance == pytest.approx(10_000.0)

    @pytest.mark.parametrize(
        "side,token_attr",
        [("yes", "yes_token_id"), ("no", "no_token_id")],
        ids=["buy-yes", "buy-no"],
    )
    def test_buy_shares(self, executor, side, token_attr):
        """buy() places a market order for YES or NO shares."""
        token_id = getattr(executor, token_attr, "tok_yes_1")
        result = executor.buy(token_id=token_id, amount_usdc=100.0)
        assert result["success"] is True

    def test_sell_shares(self, executor):
        """sell() sells shares from an existing position."""
        # Buy first to have shares
        executor.buy(token_id="tok_yes_1", amount_usdc=100.0)
        # Now sell some shares
        shares = 50.0
        result = executor.sell(token_id="tok_yes_1", amount_shares=shares)
        assert result["success"] is True

    def test_get_positions(self, executor):
        """get_positions() returns current share holdings."""
        executor.buy(token_id="tok_yes_1", amount_usdc=100.0)
        positions = executor.get_positions()
        assert len(positions) > 0

    def test_cancel_all(self, executor):
        """cancel_all() returns True on success."""
        assert executor.cancel_all() is True

    def test_dry_run_mode(self, mock_clob):
        """In dry_run mode, trades are logged but not executed."""
        with patch(
            "torchtrade.envs.live.polymarket.order_executor.ClobClient",
            return_value=mock_clob,
        ):
            from torchtrade.envs.live.polymarket.order_executor import (
                PolymarketOrderExecutor,
            )

            exe = PolymarketOrderExecutor(
                private_key="0xtest", chain_id=137, dry_run=True
            )
            exe.client = mock_clob
            initial_balance = exe.get_balance()
            result = exe.buy(token_id="tok_yes_1", amount_usdc=100.0)
            assert result["success"] is True
            assert result.get("dry_run") is True
            # Balance unchanged in dry-run
            assert exe.get_balance() == pytest.approx(initial_balance)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/envs/polymarket/test_order_executor.py -v`
Expected: FAIL with import errors (module doesn't exist yet)

- [ ] **Step 3: Write minimal implementation**

```python
# torchtrade/envs/live/polymarket/order_executor.py
"""Order executor for Polymarket CLOB trading via py-clob-client."""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)

try:
    from py_clob_client.client import ClobClient
    from py_clob_client.clob_types import MarketOrderArgs, OrderType
    from py_clob_client.order_builder.constants import BUY, SELL
except ImportError:
    ClobClient = None
    MarketOrderArgs = None
    OrderType = None
    BUY = "BUY"
    SELL = "SELL"


class PolymarketOrderExecutor:
    """Executes trades on Polymarket via the CLOB API.

    Wraps py-clob-client for buying/selling YES/NO outcome shares.
    Supports dry-run mode for paper trading.
    """

    def __init__(
        self,
        private_key: str,
        chain_id: int = 137,
        signature_type: int = 0,
        funder: Optional[str] = None,
        dry_run: bool = False,
    ):
        if ClobClient is None:
            raise ImportError(
                "py-clob-client is required. Install with: pip install py-clob-client"
            )

        self._dry_run = dry_run
        self.client = ClobClient(
            host="https://clob.polymarket.com",
            key=private_key,
            chain_id=chain_id,
            signature_type=signature_type,
            funder=funder,
        )
        self.client.set_api_creds(self.client.create_or_derive_api_creds())

    def get_balance(self) -> float:
        """Get USDC balance in dollars."""
        try:
            result = self.client.get_balance_allowance()
            raw_balance = float(result.get("balance", 0))
            return raw_balance / 1e6
        except Exception:
            logger.exception("Failed to get balance")
            return 0.0

    def buy(self, token_id: str, amount_usdc: float) -> dict:
        """Buy shares for a token.

        Args:
            token_id: The YES or NO token ID.
            amount_usdc: Dollar amount to spend.

        Returns:
            Dict with 'success' bool and order details.
        """
        if self._dry_run:
            logger.info("DRY RUN: buy %s for $%.2f", token_id, amount_usdc)
            return {"success": True, "dry_run": True}

        try:
            order_args = MarketOrderArgs(
                token_id=token_id, amount=amount_usdc, side=BUY
            )
            signed_order = self.client.create_market_order(order_args)
            result = self.client.post_order(signed_order, OrderType.FOK)
            return {"success": True, "order": result}
        except Exception as e:
            logger.error("Buy failed for token %s: %s", token_id, e)
            return {"success": False, "error": str(e)}

    def sell(self, token_id: str, amount_shares: float) -> dict:
        """Sell shares for a token.

        Args:
            token_id: The YES or NO token ID.
            amount_shares: Number of shares to sell.

        Returns:
            Dict with 'success' bool and order details.
        """
        if self._dry_run:
            logger.info("DRY RUN: sell %.4f of %s", amount_shares, token_id)
            return {"success": True, "dry_run": True}

        try:
            order_args = MarketOrderArgs(
                token_id=token_id, amount=amount_shares, side=SELL
            )
            signed_order = self.client.create_market_order(order_args)
            result = self.client.post_order(signed_order, OrderType.FOK)
            return {"success": True, "order": result}
        except Exception as e:
            logger.error("Sell failed for token %s: %s", token_id, e)
            return {"success": False, "error": str(e)}

    def get_positions(self) -> list:
        """Get current share holdings."""
        try:
            return self.client.get_positions()
        except Exception:
            logger.exception("Failed to get positions")
            return []

    def cancel_all(self) -> bool:
        """Cancel all open orders."""
        try:
            self.client.cancel_all()
            return True
        except Exception:
            logger.exception("Cancel all failed")
            return False

    def get_yes_price(self, token_id: str) -> float:
        """Get current midpoint price for a token."""
        try:
            return float(self.client.get_midpoint(token_id))
        except Exception:
            logger.exception("Failed to get price for %s", token_id)
            return 0.0

    def close_position(self, token_id: str) -> dict:
        """Close all shares for a token."""
        try:
            positions = self.client.get_positions()
            for pos in positions:
                if pos.get("asset") == token_id:
                    size = float(pos.get("size", 0))
                    if size > 0:
                        return self.sell(token_id, size)
            return {"success": True, "already_flat": True}
        except Exception as e:
            logger.error("Close position failed for %s: %s", token_id, e)
            return {"success": False, "error": str(e)}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/envs/polymarket/test_order_executor.py -v`
Expected: PASS (all 7 tests)

- [ ] **Step 5: Commit**

```bash
git add torchtrade/envs/live/polymarket/order_executor.py tests/envs/polymarket/test_order_executor.py
git commit -m "feat(polymarket): add order executor wrapping py-clob-client"
```

---

## Chunk 2: Observation Class

### Task 3: Polymarket Observation Class

**Files:**
- Create: `torchtrade/envs/live/polymarket/observation.py`
- Test: `tests/envs/polymarket/test_observation.py`

Fetches market state from the Gamma API and CLOB at each time bar. Produces a 5-element `market_state` tensor. Also exposes `is_market_closed()` for termination detection.

- [ ] **Step 1: Write the failing test**

```python
# tests/envs/polymarket/test_observation.py
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
        with patch(
            "torchtrade.envs.live.polymarket.observation.requests.get"
        ) as mock_get:
            # Mock Gamma API response for market metadata
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
            return obs

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
        # yes_price
        assert state[0] == pytest.approx(0.72, abs=0.01)
        # spread (0.73 - 0.70 = 0.03)
        assert state[1] == pytest.approx(0.03, abs=0.01)
        # volume_24h
        assert state[2] > 0
        # liquidity
        assert state[3] > 0
        # time_to_resolution (should be close to 1.0 since 30 days out)
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/envs/polymarket/test_observation.py -v`
Expected: FAIL with import errors

- [ ] **Step 3: Write minimal implementation**

```python
# torchtrade/envs/live/polymarket/observation.py
"""Polymarket observation class — fetches market state each time bar."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import requests
import torch
from torchrl.data import Bounded

from torchtrade.envs.live.polymarket.market_scanner import GAMMA_API_BASE

logger = logging.getLogger(__name__)


class PolymarketObservationClass:
    """Fetches Polymarket market state for a single market each time bar.

    Produces a 5-element market_state vector:
    [yes_price, spread, volume_24h, liquidity, time_to_resolution]
    """

    def __init__(
        self,
        yes_token_id: str,
        market_slug: str = "",
        condition_id: str = "",
        clob_client=None,
        feature_preprocessing_fn=None,
    ):
        self.yes_token_id = yes_token_id
        self.market_slug = market_slug
        self.condition_id = condition_id
        self.clob_client = clob_client
        self._feature_preprocessing_fn = feature_preprocessing_fn

        # Fetch initial market metadata from Gamma API
        self._market_meta = self._fetch_market_metadata()

        # Resolve NO token ID from metadata
        try:
            import json
            token_ids = json.loads(self._market_meta.get("clobTokenIds", "[]"))
            self.no_token_id = token_ids[1] if len(token_ids) > 1 else ""
        except (ValueError, IndexError, TypeError):
            self.no_token_id = ""

    def _fetch_market_metadata(self) -> dict:
        """Fetch market metadata from Gamma API."""
        try:
            if self.market_slug:
                resp = requests.get(
                    f"{GAMMA_API_BASE}/markets",
                    params={"slug": self.market_slug},
                    timeout=15,
                )
            elif self.condition_id:
                resp = requests.get(
                    f"{GAMMA_API_BASE}/markets",
                    params={"condition_id": self.condition_id},
                    timeout=15,
                )
            else:
                resp = requests.get(
                    f"{GAMMA_API_BASE}/markets",
                    params={"clob_token_ids": self.yes_token_id},
                    timeout=15,
                )
            resp.raise_for_status()
            data = resp.json()
            # API may return a list or a single object
            if isinstance(data, list):
                return data[0] if data else {}
            return data
        except Exception:
            logger.exception("Failed to fetch market metadata")
            return {}

    def _refresh_market_metadata(self):
        """Refresh market metadata (called each step for freshness)."""
        self._market_meta = self._fetch_market_metadata()

    def get_observations(self) -> dict:
        """Fetch current market state and return as dict.

        Returns:
            Dict with 'market_state' key -> np.ndarray of shape (5,).
        """
        self._refresh_market_metadata()

        # YES price from CLOB midpoint
        yes_price = self.get_yes_price()

        # Spread from order book
        spread = self._get_spread()

        # Volume and liquidity from Gamma metadata
        volume_24h = float(self._market_meta.get("volume24hr", 0))
        liquidity = float(self._market_meta.get("liquidity", 0))

        # Time to resolution
        time_to_resolution = self._get_time_to_resolution()

        market_state = np.array(
            [yes_price, spread, volume_24h, liquidity, time_to_resolution],
            dtype=np.float32,
        )

        if self._feature_preprocessing_fn is not None:
            market_state = self._feature_preprocessing_fn(market_state)

        return {"market_state": market_state}

    def get_observation_spec(self) -> dict:
        """Return TorchRL spec for market_state."""
        return {
            "market_state": Bounded(
                low=0.0, high=float("inf"), shape=(5,), dtype=torch.float32
            ),
        }

    def get_yes_price(self) -> float:
        """Get current YES midpoint price from CLOB."""
        if self.clob_client is not None:
            try:
                return float(self.clob_client.get_midpoint(self.yes_token_id))
            except Exception:
                pass
        # Fallback to Gamma metadata
        try:
            import json
            prices = json.loads(self._market_meta.get("outcomePrices", '["0.5","0.5"]'))
            return float(prices[0])
        except (ValueError, IndexError, TypeError):
            return 0.5

    def _get_spread(self) -> float:
        """Get bid-ask spread from CLOB order book."""
        if self.clob_client is None:
            return 0.0
        try:
            book = self.clob_client.get_order_book(self.yes_token_id)
            best_bid = float(book.bids[0].price) if book.bids else 0.0
            best_ask = float(book.asks[0].price) if book.asks else 1.0
            return best_ask - best_bid
        except Exception:
            return 0.0

    def _get_time_to_resolution(self) -> float:
        """Normalized time remaining until market resolution (1.0 → 0.0)."""
        end_date_str = self._market_meta.get("endDate", "")
        if not end_date_str:
            return 1.0
        try:
            end_dt = datetime.fromisoformat(end_date_str.replace("Z", "+00:00"))
            now = datetime.now(timezone.utc)
            remaining = (end_dt - now).total_seconds()
            if remaining <= 0:
                return 0.0
            # Normalize: assume max 365 days
            max_seconds = 365 * 24 * 3600
            return min(remaining / max_seconds, 1.0)
        except (ValueError, TypeError):
            return 1.0

    def is_market_closed(self) -> bool:
        """Check if the market has resolved/closed."""
        return bool(self._market_meta.get("closed", False))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/envs/polymarket/test_observation.py -v`
Expected: PASS (all 7 tests)

- [ ] **Step 5: Commit**

```bash
git add torchtrade/envs/live/polymarket/observation.py tests/envs/polymarket/test_observation.py
git commit -m "feat(polymarket): add observation class for market state fetching"
```

---

## Chunk 3: PolyTimeBarEnv

### Task 4: PolyTimeBarEnv — Main Environment

**Files:**
- Create: `torchtrade/envs/live/polymarket/env.py`
- Modify: `torchtrade/envs/live/polymarket/__init__.py`
- Test: `tests/envs/polymarket/test_env.py`

The main environment. Subclasses `TorchTradeLiveEnv`, uses dependency injection for observer/trader, supports supplementary observers, manages the step loop.

Reference files for pattern matching:
- `torchtrade/envs/live/alpaca/base.py` — observer/trader init, observation building, portfolio value
- `torchtrade/envs/live/alpaca/env.py` — step loop, fractional position sizing, termination
- `torchtrade/envs/core/live.py` — base class interface
- `torchtrade/envs/core/state.py` — PositionState, HistoryTracker

- [ ] **Step 1: Write the failing test**

```python
# tests/envs/polymarket/test_env.py
"""Tests for PolyTimeBarEnv."""

import pytest
import torch
from tensordict import TensorDict

from tests.envs.polymarket.mocks import MockPolymarketObserver, MockPolymarketTrader


class TestPolyTimeBarEnv:
    """Tests for PolyTimeBarEnv."""

    @pytest.fixture
    def env(self):
        from torchtrade.envs.live.polymarket.env import (
            PolyTimeBarEnv,
            PolyTimeBarEnvConfig,
        )

        config = PolyTimeBarEnvConfig(
            yes_token_id="tok_yes_1",
            execute_on="1Hour",
            action_levels=[-1, 0, 1],
            initial_cash=10_000.0,
        )
        observer = MockPolymarketObserver(yes_price=0.72)
        trader = MockPolymarketTrader(initial_balance=10_000.0, yes_price=0.72)
        env = PolyTimeBarEnv(
            config=config,
            observer=observer,
            trader=trader,
        )
        return env

    def test_observation_spec_has_required_keys(self, env):
        """observation_spec includes market_state and account_state."""
        spec = env.observation_spec
        assert "market_state" in spec.keys()
        assert "account_state" in spec.keys()

    def test_action_spec(self, env):
        """Action spec matches action_levels count."""
        assert env.action_spec.n == 3

    def test_reset_returns_valid_tensordict(self, env):
        """reset() returns TensorDict with market_state and account_state."""
        td = env.reset()
        assert "market_state" in td.keys()
        assert "account_state" in td.keys()
        assert td["market_state"].shape == (5,)
        assert td["account_state"].shape == (6,)

    def test_account_state_initial_values(self, env):
        """After reset, account state shows flat position."""
        td = env.reset()
        acct = td["account_state"]
        assert acct[0].item() == pytest.approx(0.0)  # exposure_pct
        assert acct[1].item() == pytest.approx(0.0)  # position_direction
        assert acct[4].item() == pytest.approx(1.0)  # leverage
        assert acct[5].item() == pytest.approx(1.0)  # distance_to_liquidation

    @pytest.mark.parametrize(
        "action_idx,expected_direction",
        [
            (0, -1),  # action_levels[-1] -> short/NO
            (1, 0),   # action_levels[0]  -> flat
            (2, 1),   # action_levels[1]  -> long/YES
        ],
        ids=["buy-no", "flat", "buy-yes"],
    )
    def test_step_action_direction(self, env, action_idx, expected_direction):
        """Step with different actions produces correct position direction."""
        env.reset()
        td_in = TensorDict({"action": torch.tensor(action_idx)}, batch_size=())
        td_out = env._step(td_in)
        assert "reward" in td_out.keys()
        assert "done" in td_out.keys()
        assert "terminated" in td_out.keys()
        # Check position direction matches
        direction = td_out["account_state"][1].item()
        if expected_direction != 0:
            assert direction == pytest.approx(expected_direction, abs=0.1)
        else:
            assert direction == pytest.approx(0.0, abs=0.1)

    def test_step_flat_no_trade(self, env):
        """action=flat when already flat produces no trade."""
        env.reset()
        initial_balance = env.trader.get_balance()
        td_in = TensorDict({"action": torch.tensor(1)}, batch_size=())  # 0 -> flat
        env._step(td_in)
        assert env.trader.get_balance() == pytest.approx(initial_balance)

    def test_termination_on_market_close(self, env):
        """Episode terminates when market resolves."""
        env.reset()
        env.observer.market_closed = True
        td_in = TensorDict({"action": torch.tensor(1)}, batch_size=())
        td_out = env._step(td_in)
        assert td_out["terminated"].item() is True

    def test_supplementary_observers_merged(self):
        """Supplementary observer specs and observations are merged."""
        from torchtrade.envs.live.polymarket.env import (
            PolyTimeBarEnv,
            PolyTimeBarEnvConfig,
        )
        import numpy as np
        from torchrl.data import Bounded

        config = PolyTimeBarEnvConfig(
            yes_token_id="tok_yes_1",
            action_levels=[-1, 0, 1],
        )

        # Create a mock supplementary observer
        class FakeSupplementary:
            def get_observation_spec(self):
                return {
                    "extra_data": Bounded(
                        low=-torch.inf, high=torch.inf, shape=(3,), dtype=torch.float32
                    )
                }

            def get_observations(self):
                return {"extra_data": np.array([1.0, 2.0, 3.0], dtype=np.float32)}

        env = PolyTimeBarEnv(
            config=config,
            observer=MockPolymarketObserver(),
            trader=MockPolymarketTrader(),
            supplementary_observers=[FakeSupplementary()],
        )
        assert "extra_data" in env.observation_spec.keys()
        td = env.reset()
        assert "extra_data" in td.keys()
        assert td["extra_data"].shape == (3,)

    def test_supplementary_key_collision_raises(self):
        """Key collision between supplementary observers raises ValueError."""
        from torchtrade.envs.live.polymarket.env import (
            PolyTimeBarEnv,
            PolyTimeBarEnvConfig,
        )
        import numpy as np
        from torchrl.data import Bounded

        config = PolyTimeBarEnvConfig(
            yes_token_id="tok_yes_1",
            action_levels=[-1, 0, 1],
        )

        class CollidingObserver:
            def get_observation_spec(self):
                return {
                    "market_state": Bounded(
                        low=0, high=1, shape=(5,), dtype=torch.float32
                    )
                }

            def get_observations(self):
                return {"market_state": np.zeros(5, dtype=np.float32)}

        with pytest.raises(ValueError, match="collision"):
            PolyTimeBarEnv(
                config=config,
                observer=MockPolymarketObserver(),
                trader=MockPolymarketTrader(),
                supplementary_observers=[CollidingObserver()],
            )

    def test_max_steps_truncation(self):
        """Episode truncates after max_steps."""
        from torchtrade.envs.live.polymarket.env import (
            PolyTimeBarEnv,
            PolyTimeBarEnvConfig,
        )

        config = PolyTimeBarEnvConfig(
            yes_token_id="tok_yes_1",
            action_levels=[-1, 0, 1],
            max_steps=2,
        )
        env = PolyTimeBarEnv(
            config=config,
            observer=MockPolymarketObserver(),
            trader=MockPolymarketTrader(),
        )
        env.reset()
        td1 = env._step(TensorDict({"action": torch.tensor(1)}, batch_size=()))
        assert td1["truncated"].item() is False
        td2 = env._step(TensorDict({"action": torch.tensor(1)}, batch_size=()))
        assert td2["truncated"].item() is True

    def test_bankruptcy_termination(self):
        """Episode terminates when balance drops below threshold."""
        from torchtrade.envs.live.polymarket.env import (
            PolyTimeBarEnv,
            PolyTimeBarEnvConfig,
        )

        config = PolyTimeBarEnvConfig(
            yes_token_id="tok_yes_1",
            action_levels=[-1, 0, 1],
            done_on_bankruptcy=True,
            bankrupt_threshold=0.5,
        )
        # Start with very low balance to trigger bankruptcy
        trader = MockPolymarketTrader(initial_balance=1.0, yes_price=0.72)
        env = PolyTimeBarEnv(
            config=config,
            observer=MockPolymarketObserver(),
            trader=trader,
        )
        env.reset()
        env._initial_balance = 10_000.0  # Pretend we started rich
        td = env._step(TensorDict({"action": torch.tensor(1)}, batch_size=()))
        assert td["terminated"].item() is True

    def test_missing_market_id_raises(self):
        """ValueError raised when no market identifier is provided."""
        from torchtrade.envs.live.polymarket.env import (
            PolyTimeBarEnv,
            PolyTimeBarEnvConfig,
        )

        config = PolyTimeBarEnvConfig()  # All empty
        with pytest.raises(ValueError, match="market identifier"):
            PolyTimeBarEnv(
                config=config,
                observer=MockPolymarketObserver(),
                trader=MockPolymarketTrader(),
            )

    def test_close_position_on_reset(self):
        """When close_position_on_reset=True, positions are closed on reset."""
        from torchtrade.envs.live.polymarket.env import (
            PolyTimeBarEnv,
            PolyTimeBarEnvConfig,
        )

        config = PolyTimeBarEnvConfig(
            yes_token_id="tok_yes_1",
            action_levels=[-1, 0, 1],
            close_position_on_reset=True,
            close_position_on_init=False,
        )
        trader = MockPolymarketTrader(initial_balance=10_000.0)
        env = PolyTimeBarEnv(
            config=config,
            observer=MockPolymarketObserver(),
            trader=trader,
        )
        env.reset()
        # Buy YES
        env._step(TensorDict({"action": torch.tensor(2)}, batch_size=()))
        # Reset should close position
        env.reset()
        assert env.position.current_position == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/envs/polymarket/test_env.py -v`
Expected: FAIL with import errors

- [ ] **Step 3: Write minimal implementation**

```python
# torchtrade/envs/live/polymarket/env.py
"""PolyTimeBarEnv — Polymarket prediction market TorchRL trading environment."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Protocol, Union, runtime_checkable

import torch
from tensordict import TensorDict, TensorDictBase
from torchrl.data import Bounded, Categorical
from torchrl.data.tensor_specs import Composite

from torchtrade.envs.core.default_rewards import log_return_reward
from torchtrade.envs.core.live import TorchTradeLiveEnv
from torchtrade.envs.core.state import HistoryTracker, PositionState

logger = logging.getLogger(__name__)


@runtime_checkable
class SupplementaryObserver(Protocol):
    """Protocol for supplementary data sources (e.g., Binance OHLCV)."""

    def get_observation_spec(self) -> dict:
        """Return dict mapping key names to TorchRL TensorSpec."""
        ...

    def get_observations(self) -> dict:
        """Return current observations as dict of arrays/tensors."""
        ...


# Map execute_on strings like "1Hour", "5Minute", "1Day" to (value, unit)
_EXECUTE_ON_PATTERN = re.compile(r"^(\d+)(Minute|Hour|Day)$")


@dataclass
class PolyTimeBarEnvConfig:
    """Configuration for PolyTimeBarEnv."""

    # Market identification — priority: yes_token_id > condition_id > market_slug
    # At least one must be non-empty
    market_slug: str = ""
    condition_id: str = ""
    yes_token_id: str = ""

    # Stepping
    execute_on: str = "1Hour"
    max_steps: Optional[int] = None

    # Actions
    action_levels: List[float] = field(default_factory=lambda: [-1, 0, 1])

    # Capital
    initial_cash: float = 10_000.0

    # Termination
    done_on_bankruptcy: bool = True
    bankrupt_threshold: float = 0.1

    # Position management
    close_position_on_init: bool = True
    close_position_on_reset: bool = False

    # Mode
    dry_run: bool = False

    seed: Optional[int] = 42


class PolyTimeBarEnv(TorchTradeLiveEnv):
    """TorchRL environment for Polymarket prediction market trading.

    Steps on regular time bars. Manages a single market's YES/NO position.
    Supports supplementary observers for augmenting observations with
    external data sources (e.g., Binance OHLCV).
    """

    ACCOUNT_STATE_LABELS = [
        "exposure_pct",
        "position_direction",
        "unrealized_pnl_pct",
        "holding_time",
        "leverage",
        "distance_to_liquidation",
    ]

    def __init__(
        self,
        config: PolyTimeBarEnvConfig,
        private_key: str = "",
        observer=None,
        trader=None,
        supplementary_observers: Optional[List[SupplementaryObserver]] = None,
        reward_function: Optional[Callable] = None,
        feature_preprocessing_fn: Optional[Callable] = None,
    ):
        # Validate market identification
        if not (config.yes_token_id or config.condition_id or config.market_slug):
            raise ValueError(
                "At least one market identifier must be provided: "
                "yes_token_id, condition_id, or market_slug"
            )

        self._private_key = private_key
        self._supplementary_observers = supplementary_observers or []
        self.reward_function = reward_function or log_return_reward
        self._feature_preprocessing_fn = feature_preprocessing_fn

        self.action_levels = config.action_levels
        self.history = HistoryTracker()
        self._step_count = 0
        self._initial_balance = config.initial_cash

        # Call parent — passes empty strings for api_key/api_secret
        # Our _init_trading_clients override uses self._private_key instead
        super().__init__(
            config=config,
            api_key="",
            api_secret="",
            observer=observer,
            trader=trader,
            timezone="UTC",
        )

        # Parse execute_on into value + unit for _wait_for_next_timestamp()
        match = _EXECUTE_ON_PATTERN.match(config.execute_on)
        if match:
            self.execute_on_value = int(match.group(1))
            self.execute_on_unit = match.group(2)
        else:
            self.execute_on_value = 1
            self.execute_on_unit = "Hour"

        # Resolve token IDs from observer (observer resolves from Gamma API)
        self._yes_token_id = config.yes_token_id or getattr(self.observer, "yes_token_id", "")
        self._no_token_id = getattr(self.observer, "no_token_id", "")

        # Build specs after parent init (observer/trader are now set)
        self._build_observation_specs()
        self.action_spec = Categorical(len(self.action_levels))

        # Close position on init if configured
        if config.close_position_on_init and self.trader is not None:
            self.trader.cancel_all()
            if self._yes_token_id:
                self.trader.close_position(self._yes_token_id)
            if self._no_token_id:
                self.trader.close_position(self._no_token_id)

    def _init_trading_clients(self, api_key, api_secret, observer, trader):
        """Initialize observer and trader using private_key (ignores api_key/api_secret)."""
        self.observer = observer
        self.trader = trader

        if self.observer is None:
            from torchtrade.envs.live.polymarket.observation import (
                PolymarketObservationClass,
            )

            self.observer = PolymarketObservationClass(
                yes_token_id=self.config.yes_token_id,
                market_slug=self.config.market_slug,
                condition_id=self.config.condition_id,
                feature_preprocessing_fn=self._feature_preprocessing_fn,
            )

        if self.trader is None:
            from torchtrade.envs.live.polymarket.order_executor import (
                PolymarketOrderExecutor,
            )

            self.trader = PolymarketOrderExecutor(
                private_key=self._private_key,
                dry_run=self.config.dry_run,
            )

    def _build_observation_specs(self):
        """Build observation spec from market_state + account_state + supplementary."""
        self.observation_spec = Composite(shape=())

        # Market state (5,)
        self.observation_spec.set(
            "market_state",
            Bounded(low=0.0, high=float("inf"), shape=(5,), dtype=torch.float32),
        )

        # Account state (6,) — standard across all TorchTrade envs
        self.observation_spec.set(
            "account_state",
            Bounded(
                low=-float("inf"), high=float("inf"), shape=(6,), dtype=torch.float32
            ),
        )

        # Supplementary observer specs
        reserved_keys = set(self.observation_spec.keys())
        for supp in self._supplementary_observers:
            for key, spec in supp.get_observation_spec().items():
                if key in reserved_keys:
                    raise ValueError(
                        f"Supplementary observer key collision: '{key}' "
                        f"already exists in observation spec"
                    )
                self.observation_spec.set(key, spec)
                reserved_keys.add(key)

    def _get_observation(self) -> TensorDictBase:
        """Build observation TensorDict for current time bar."""
        # Market state from observer
        obs = self.observer.get_observations()
        market_state = torch.tensor(obs["market_state"], dtype=torch.float32)

        # Cache yes_price from the observation to avoid redundant CLOB call
        self._last_yes_price = float(market_state[0])

        # Account state
        account_state = self._build_account_state(self._last_yes_price)

        td = TensorDict(
            {"market_state": market_state, "account_state": account_state},
            batch_size=(),
        )

        # Supplementary observations
        for supp in self._supplementary_observers:
            for key, val in supp.get_observations().items():
                td.set(key, torch.tensor(val, dtype=torch.float32))

        return td

    def _build_account_state(self, current_price: float) -> torch.Tensor:
        """Build 6-element account state tensor."""
        portfolio_value = self._get_portfolio_value()

        if portfolio_value > 0 and self.position.position_value > 0:
            exposure_pct = self.position.position_value / portfolio_value
        else:
            exposure_pct = 0.0

        # Unrealized PnL
        if self.position.entry_price > 0 and self.position.current_position != 0:
            unrealized_pnl = (
                (current_price - self.position.entry_price)
                / self.position.entry_price
                * self.position.current_position
            )
        else:
            unrealized_pnl = 0.0

        return torch.tensor(
            [
                exposure_pct,
                self.position.current_position,
                unrealized_pnl,
                float(self.position.hold_counter),
                1.0,  # leverage (always 1.0)
                1.0,  # distance_to_liquidation (always 1.0)
            ],
            dtype=torch.float32,
        )

    def _get_portfolio_value(self) -> float:
        """Calculate total portfolio value: USDC balance + position market value."""
        cash = self.trader.get_balance()
        current_price = getattr(self, "_last_yes_price", 0.0) or self.observer.get_yes_price()
        position_value = abs(self.position.position_size) * current_price
        return cash + position_value

    def _reset(self, tensordict: TensorDictBase, **kwargs) -> TensorDictBase:
        """Reset: clear history, optionally close positions, return initial obs."""
        self.history.reset()
        self.position.reset()
        self._step_count = 0
        self._last_yes_price = 0.0

        if self.config.close_position_on_reset and self.trader is not None:
            self.trader.cancel_all()
            if self._yes_token_id:
                self.trader.close_position(self._yes_token_id)
            if self._no_token_id:
                self.trader.close_position(self._no_token_id)

        obs = self._get_observation()
        self._initial_balance = self._get_portfolio_value()
        return obs

    def _step(self, tensordict: TensorDictBase) -> TensorDictBase:
        """Execute one step: process action, wait for next bar, return obs."""
        action_idx = tensordict.get("action", torch.tensor(0))
        if isinstance(action_idx, torch.Tensor):
            action_idx = action_idx.item()

        desired_action = self.action_levels[action_idx]
        yes_price = self.observer.get_yes_price()
        self._last_yes_price = yes_price

        # Execute trade
        self._execute_trade_if_needed(desired_action, yes_price)

        # Wait for next time bar (blocking — same as all other live envs)
        self._wait_for_next_timestamp()

        # Update position hold counter
        if self.position.current_position != 0:
            self.position.hold_counter += 1
        else:
            self.position.hold_counter = 0

        self._step_count += 1

        # Get updated observation (refreshes prices from exchange)
        td_out = self._get_observation()

        # Get updated portfolio value
        new_portfolio_value = self._get_portfolio_value()

        # Record history
        action_label = "hold"
        if desired_action > 0:
            action_label = "buy_yes"
        elif desired_action < 0:
            action_label = "buy_no"

        self.history.record_step(
            price=self._last_yes_price,
            action=desired_action,
            reward=0.0,
            portfolio_value=new_portfolio_value,
            position=self.position.current_position,
            action_type=action_label,
        )

        # Calculate reward
        reward = float(self.reward_function(self.history))
        self.history.rewards[-1] = reward

        # Check termination
        terminated = False
        truncated = False

        if self.observer.is_market_closed():
            terminated = True

        if (
            self.config.done_on_bankruptcy
            and self._initial_balance > 0
            and new_portfolio_value
            < self.config.bankrupt_threshold * self._initial_balance
        ):
            terminated = True

        if (
            self.config.max_steps is not None
            and self._step_count >= self.config.max_steps
        ):
            truncated = True

        done = terminated or truncated

        td_out.set("reward", torch.tensor([reward], dtype=torch.float32))
        td_out.set("done", torch.tensor([done], dtype=torch.bool))
        td_out.set("terminated", torch.tensor([terminated], dtype=torch.bool))
        td_out.set("truncated", torch.tensor([truncated], dtype=torch.bool))

        return td_out

    def _execute_trade_if_needed(self, desired_action: float, current_price: float):
        """Execute trade based on desired fractional action.

        Action mapping (portfolio-fraction, Alpaca pattern):
        - positive -> allocate fraction to YES position
        - negative -> allocate fraction to NO position
        - zero -> go flat
        """
        current_direction = self.position.current_position

        if desired_action == 0.0:
            if current_direction != 0:
                token_id = (
                    self._yes_token_id if current_direction > 0 else self._no_token_id
                )
                self.trader.close_position(token_id)
                self.position.reset()
            return

        target_direction = 1.0 if desired_action > 0 else -1.0
        target_fraction = abs(desired_action)

        # If direction changed, close existing position first
        if current_direction != 0 and current_direction != target_direction:
            token_id = (
                self._yes_token_id if current_direction > 0 else self._no_token_id
            )
            self.trader.close_position(token_id)
            self.position.reset()

        # Calculate target position value
        portfolio_value = self._get_portfolio_value()
        target_value = portfolio_value * target_fraction
        current_value = abs(self.position.position_value)

        delta_value = target_value - current_value
        if delta_value < 1.0:
            return

        # Buy the appropriate token
        token_id = self._yes_token_id if target_direction > 0 else self._no_token_id
        result = self.trader.buy(token_id=token_id, amount_usdc=delta_value)

        if result.get("success"):
            shares_bought = delta_value / current_price if current_price > 0 else 0
            self.position.current_position = target_direction
            self.position.position_size += shares_bought
            self.position.position_value = (
                abs(self.position.position_size) * current_price
            )
            if self.position.entry_price == 0:
                self.position.entry_price = current_price
            self.position.current_action_level = desired_action

    def close(self):
        """Clean up: cancel orders and warn about open positions."""
        if self.trader is not None:
            self.trader.cancel_all()
            if self.position.current_position != 0:
                logger.warning(
                    "PolyTimeBarEnv closed with open position (direction=%.0f). "
                    "Position was NOT automatically closed.",
                    self.position.current_position,
                )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/envs/polymarket/test_env.py -v`
Expected: PASS (all 10 tests)

- [ ] **Step 5: Update `__init__.py`**

```python
# torchtrade/envs/live/polymarket/__init__.py
"""Polymarket prediction market live trading environment."""

from torchtrade.envs.live.polymarket.env import PolyTimeBarEnv, PolyTimeBarEnvConfig
from torchtrade.envs.live.polymarket.market_scanner import (
    MarketScanner,
    MarketScannerConfig,
    PolymarketMarket,
)
from torchtrade.envs.live.polymarket.observation import PolymarketObservationClass
from torchtrade.envs.live.polymarket.order_executor import PolymarketOrderExecutor

__all__ = [
    "MarketScanner",
    "MarketScannerConfig",
    "PolymarketMarket",
    "PolymarketObservationClass",
    "PolymarketOrderExecutor",
    "PolyTimeBarEnv",
    "PolyTimeBarEnvConfig",
]
```

- [ ] **Step 6: Run full Polymarket test suite**

Run: `uv run pytest tests/envs/polymarket/ -v`
Expected: ALL PASS (19 scanner + 7 executor + 7 observation + 14 env = 47 tests)

- [ ] **Step 7: Commit**

```bash
git add torchtrade/envs/live/polymarket/env.py torchtrade/envs/live/polymarket/__init__.py tests/envs/polymarket/test_env.py
git commit -m "feat(polymarket): add PolyTimeBarEnv with supplementary observer support"
```

---

## Chunk 4: Cleanup

### Task 5: Remove Old Planning Files and Dead Code

- [ ] **Step 1: Delete old `planning.md`**

The original `planning.md` from the previous approach (LLM-first, market-by-market stepping) is now superseded by the spec + plan in `docs/superpowers/`.

```bash
rm planning.md
```

- [ ] **Step 2: Verify no old code from previous planning remains**

Check that no `research.py` or `polymarket_llm_actor.py` files were created from the old plan:

```bash
# Should return nothing — these files were never implemented
ls torchtrade/envs/live/polymarket/research.py 2>/dev/null
ls torchtrade/actor/polymarket_llm_actor.py 2>/dev/null
```

- [ ] **Step 3: Run full test suite to ensure nothing is broken**

Run: `uv run pytest tests/ -v`
Expected: ALL PASS

- [ ] **Step 4: Commit cleanup**

```bash
git rm planning.md
git commit -m "chore: remove old polymarket planning.md (superseded by spec + plan)"
```
