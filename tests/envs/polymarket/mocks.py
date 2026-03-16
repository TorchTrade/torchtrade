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
