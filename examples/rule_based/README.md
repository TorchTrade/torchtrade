# Rule-Based Actor Examples

Examples using the `MeanReversionActor` (Bollinger Bands + Stochastic RSI) for trading.

## Examples

### Offline Backtesting

Run the mean reversion strategy on historical BTC/USD data:

```bash
python examples/rule_based/offline.py
```

### Live Trading (Alpaca Paper)

Run the strategy on Alpaca's paper trading API:

```bash
python examples/rule_based/live.py
```

Requires `ALPACA_API_KEY` and `ALPACA_SECRET_KEY` in `.env`.

## Strategy

The `MeanReversionActor` uses:
- **Bollinger Bands** to identify overbought/oversold conditions
- **Stochastic RSI** crossovers for entry confirmation
- **Volume confirmation** (1.5x average volume required)

**Buy**: Price below lower BB + bullish Stoch RSI crossover from oversold + volume confirmed
**Sell**: Price above upper BB + bearish Stoch RSI crossover from overbought + volume confirmed
**Hold**: Otherwise

## Customization

Adjust strategy parameters when creating the actor:

```python
actor = MeanReversionActor(
    bb_window=20,           # Bollinger Bands period
    bb_std=2.0,             # BB standard deviations
    stoch_rsi_window=14,    # Stochastic RSI period
    oversold_threshold=20,  # Buy threshold
    overbought_threshold=80,# Sell threshold
)
```
