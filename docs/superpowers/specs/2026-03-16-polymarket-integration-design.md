# Polymarket Integration Design Spec

## Goal

Add Polymarket as a live trading venue in TorchTrade. The environment trades YES/NO outcome shares on a single prediction market, stepping on regular time bars like existing TorchTrade environments. Users can optionally attach supplementary data sources (e.g., Binance OHLCV) to augment observations.

Two environment variants:
1. **PolyTimeBarEnv** — time-bar stepping, supports supplementary observers, primary focus of this spec
2. **PolyMarketStepEnv** — iterates through a set of markets per step (future work, not in this spec)

## Architecture

Follows the existing Observer/Trader dependency injection pattern:

```
Observer (observation.py)     Trader (order_executor.py)
        \                          /
         \                        /
          PolyTimeBarEnv (env.py)
                |
    [optional: supplementary_observers]
```

Single market per env instance. The user configures which market (by slug, condition ID, or token ID) and the time bar interval.

### Authentication

Polymarket uses Ethereum wallet authentication (private key), not the `api_key`/`api_secret` pattern used by other exchanges. The env constructor accepts a `private_key` parameter. When observer and trader are not injected, the env creates them using this key.

When observer/trader ARE injected (e.g., for testing), `private_key` is not required.

### Base Class Integration

`TorchTradeLiveEnv.__init__` accepts `api_key`/`api_secret` and passes them to `_init_trading_clients()`. Since Polymarket uses a single `private_key` instead, `PolyTimeBarEnv` stores `private_key` on `self` before calling `super().__init__`, passing empty strings for `api_key`/`api_secret`. The overridden `_init_trading_clients()` ignores the `api_key`/`api_secret` params and uses `self._private_key` instead. This follows the same pattern Bybit/Binance use (stashing config state before `super().__init__`).

### Timezone

Polymarket is blockchain-based and operates globally. The env passes `timezone="UTC"` to `TorchTradeLiveEnv.__init__`.

## Observation Space

### Polymarket Market State (always present)

Shape: `(5,)` tensor key: `"market_state"`

| Index | Field | Description | Range |
|-------|-------|-------------|-------|
| 0 | `yes_price` | Current YES share price | 0.0–1.0 |
| 1 | `spread` | Bid-ask spread | 0.0–1.0 |
| 2 | `volume_24h` | Rolling 24h volume (normalized) | 0.0+ |
| 3 | `liquidity` | Current order book depth (normalized) | 0.0+ |
| 4 | `time_to_resolution` | Normalized time remaining | 1.0 → 0.0 |

Note: `no_price` is omitted because `no_price = 1 - yes_price` for binary markets. Including it would be a redundant feature. (The `MarketScanner` dataclass retains both prices for convenience — the observation space is what gets optimized.)

### Account State (standard 6-element)

Shape: `(6,)` tensor key: `"account_state"`

Same universal format as all TorchTrade envs:

| Index | Field | Polymarket behavior |
|-------|-------|-------------------|
| 0 | `exposure_pct` | position_value / portfolio_value |
| 1 | `position_direction` | +1 = YES, -1 = NO, 0 = flat |
| 2 | `unrealized_pnl_pct` | (current_price - entry_price) / entry_price * direction |
| 3 | `holding_time` | Steps since position opened |
| 4 | `leverage` | Always 1.0 (no leverage on Polymarket) |
| 5 | `distance_to_liquidation` | Always 1.0 (no liquidation on Polymarket) |

### Supplementary Observer Protocol (optional)

Users can attach additional observers that provide extra TensorDict keys. Example: a Binance OHLCV observer adds `market_data_1Hour_48` with shape `(48, num_features)`.

**Interface.** A supplementary observer must implement:

```python
class SupplementaryObserver(Protocol):
    def get_observation_spec(self) -> dict[str, TensorSpec]:
        """Return a dict mapping key names to their TorchRL TensorSpec.
        Called once at env construction to build the composite observation_spec."""
        ...

    def get_observations(self) -> dict[str, np.ndarray | torch.Tensor]:
        """Return current observations. Called at each step and reset."""
        ...
```

**Spec merging.** At construction, the env iterates `supplementary_observers`, calls `get_observation_spec()` on each, and adds the returned specs to the env's `Composite` observation spec. Key collisions raise a `ValueError` at construction time.

**Call timing.** `get_observations()` is called on every `_reset()` and `_step()`, alongside the primary Polymarket observer.

## Action Space

Discrete categorical using `action_levels`, identical to existing envs.

Default: `action_levels = [-1, 0, 1]`

| Action level | Meaning |
|-------------|---------|
| -1 | Full allocation to NO shares |
| 0 | Flat (no position) |
| +1 | Full allocation to YES shares |

Finer granularity supported: e.g., `[-1, -0.5, 0, 0.5, 1]`.

### Mapping to Polymarket trades

- Positive action → buy YES shares (close NO position first if needed)
- Negative action → buy NO shares (close YES position first if needed)
- Zero → close any open position

Position sizing follows the Alpaca pattern (portfolio-fraction allocation, no leverage). The agent's action level represents the target fraction of portfolio allocated to the position.

## Trade Execution

`PolymarketOrderExecutor` wraps `py-clob-client`:

- `buy(token_id, amount_usdc)` → market buy YES or NO shares
- `sell(token_id, amount_shares)` → market sell shares
- `get_balance()` → USDC balance
- `get_positions()` → current share holdings
- `get_order_book(token_id)` → order book for slippage estimation
- `cancel_all()` → cancel open orders

Orders are Fill-or-Kill (FOK) market orders via the Polymarket CLOB.

When `dry_run=True` in config, the executor logs intended trades but does not submit them to the CLOB. Balance and position tracking are simulated locally.

## Observation Class

`PolymarketObservationClass` fetches market state each time bar:

- Current YES price (midpoint from CLOB)
- Spread (best ask - best bid)
- 24h volume and liquidity from Gamma API
- Time to resolution (computed from market end date)
- Market status (active/closed/resolved) — used for episode termination

Implements the same observer interface as other live exchanges: `get_observations() -> dict`.

An optional `feature_preprocessing_fn` can be passed to the env constructor (consistent with all other live envs) for custom feature engineering on the market state.

### Observation Spec Construction

Unlike other live envs that build observation specs from `time_frames`/`window_sizes` config, `PolyTimeBarEnv` builds its spec from:

1. **`market_state`**: Hardcoded `Bounded` spec with shape `(5,)` — always present.
2. **`account_state`**: Hardcoded `Bounded` spec with shape `(6,)` — always present (standard across all TorchTrade envs).
3. **Supplementary specs**: Iterated from `supplementary_observers[i].get_observation_spec()` at construction.

No `time_frames`, `window_sizes`, or `include_base_features` config fields are needed — Polymarket's primary observation is a fixed-shape market state vector, not multi-timeframe OHLCV windows. Supplementary observers that provide OHLCV data carry their own timeframe/window config internally.

## Reward

Default: `log_return_reward` based on portfolio value changes (same as all TorchTrade envs).

Share prices move between 0 and 1 continuously. On resolution, shares pay $1 (if outcome is YES) or $0 (if outcome is NO). From the reward function's perspective, resolution is just a final price move.

Custom reward functions are injected via the env constructor parameter `reward_function` (not in config), consistent with all other live envs.

## Episode Termination

| Condition | Detection mechanism | Type |
|-----------|-------------------|------|
| Market resolves | Observer checks `market.closed` flag from Gamma API each step. If closed, query resolution outcome, compute final portfolio value, then terminate. | `terminated = True` |
| Balance below bankruptcy threshold | `balance < bankrupt_threshold * initial_balance` | `terminated = True` |
| Max steps reached | Step counter exceeds `max_steps` config | `truncated = True` |

## Configuration

```python
@dataclass
class PolyTimeBarEnvConfig:
    # Market identification — priority: yes_token_id > condition_id > market_slug
    # At least one must be non-empty; env raises ValueError at construction if none provided
    market_slug: str = ""                        # Polymarket market slug (resolved to token ID via Gamma API)
    condition_id: str = ""                       # Condition ID (resolved to token ID via Gamma API)
    yes_token_id: str = ""                       # Direct YES token ID (no resolution needed)

    # Stepping
    execute_on: Union[str, TimeFrame] = "1Hour"  # Time bar interval
    max_steps: Optional[int] = None              # Max steps before truncation

    # Actions
    action_levels: List[float] = field(default_factory=lambda: [-1, 0, 1])

    # Capital
    initial_cash: float = 10_000.0               # USDC

    # Termination
    done_on_bankruptcy: bool = True
    bankrupt_threshold: float = 0.1

    # Position management
    close_position_on_init: bool = True
    close_position_on_reset: bool = False

    # Mode
    dry_run: bool = False                        # Log trades without submitting

    seed: Optional[int] = 42
```

## Constructor

```python
class PolyTimeBarEnv(TorchTradeLiveEnv):
    def __init__(
        self,
        config: PolyTimeBarEnvConfig,
        private_key: str = "",                                    # Ethereum wallet key
        observer: Optional[PolymarketObservationClass] = None,    # Injected or auto-created
        trader: Optional[PolymarketOrderExecutor] = None,         # Injected or auto-created
        supplementary_observers: Optional[List[SupplementaryObserver]] = None,
        reward_function: Optional[Callable] = None,               # Default: log_return_reward
        feature_preprocessing_fn: Optional[Callable] = None,
    ):
```

## Usage Example

```python
from torchtrade.envs.live.polymarket import (
    PolyTimeBarEnv,
    PolyTimeBarEnvConfig,
)

config = PolyTimeBarEnvConfig(
    market_slug="will-bitcoin-exceed-100k",
    execute_on="1Hour",
    action_levels=[-1, 0, 1],
)

# Minimal setup — env creates observer/trader from private_key
env = PolyTimeBarEnv(config=config, private_key="0x...")

# With supplementary Binance OHLCV observer
env = PolyTimeBarEnv(
    config=config,
    private_key="0x...",
    supplementary_observers=[binance_btc_observer],
)

# Works with any TorchRL-compatible policy
td = env.reset()
td = policy(td)
td = env.step(td)
```

## File Structure

```
torchtrade/envs/live/polymarket/
├── __init__.py
├── market_scanner.py          # Already implemented
├── observation.py             # Polymarket price/state observer
├── order_executor.py          # py-clob-client wrapper
└── env.py                     # PolyTimeBarEnv

tests/envs/polymarket/
├── __init__.py
├── test_market_scanner.py     # Already implemented
├── mocks.py                   # Mock Polymarket API responses
├── test_observation.py
├── test_order_executor.py
└── test_env.py
```

## What This Spec Does NOT Cover

- **PolyMarketStepEnv** — the market-by-market stepping variant. Separate future spec.
- **LLM actors / research pipelines** — policy-level concerns, not env concerns.
- **Offline backtesting** — would require historical Polymarket price data. Future work.
- **Multi-market portfolios** — one market per env instance for now.

## Dependencies

- `py-clob-client` — Polymarket CLOB API client
- `requests` — Gamma API queries (market metadata)
- Existing TorchTrade: `TorchTradeLiveEnv`, `PositionState`, `HistoryTracker`
