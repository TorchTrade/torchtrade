# LLM Finetuning on traces with natural language for decision making 


## Idea:

Currently the hard coded strategy (trend) creates a decision making summary after each step. 

```bash
Decision summary:
Timeframe signals: {'1m': 'neutral', '5m': 'neutral', '15m': 'neutral', '1h': 'neutral'}
Bullish TFs: 0, Bearish TFs: 0
Volatility filter (5m): Failed (vol=0.000302)
BB % (5m): 0.45, Volume Z: -0.36, VWAP Dev: 1711.283936, ATR: 0.000667
Candle (5m): Bullish=No, Bearish=Yes
Action: Hold (entry conditions not met)
----------
```

This summary along with the current state of the environment or other information could be used to finetune a LLM to make decisions.