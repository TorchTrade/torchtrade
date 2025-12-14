# SeqLongOnlyEnv
Environment for sequential long-only trading.
Used to train agents for the alpaca crpyto trading as there is long only actions.

## Observation Spec
Flexible observation spec that can be defined by the user via the time_frames, window_sizes and feature_preprocessing_fn.
Utilizes the `MarketDataObservationSampler` to sample market data single or multi time frame. 


## Action Spec
Agent can take three actions: 0, 1, 2. which translate to [-1.0, 0.0, 1.0]  # Sell-all, Do-Nothing, Buy-all. Agent always buys with all possible cash, does nothing or sells the open position. 

## Config 

```python

class SeqLongOnlyEnvConfig:
    symbol: str = "BTC/USD"
    time_frames: Union[List[TimeFrame], TimeFrame] = TimeFrame(1, TimeFrameUnit.Minute)
    window_sizes: Union[List[int], int] = 10
    execute_on: TimeFrame = TimeFrame(1, TimeFrameUnit.Minute) # On which timeframe to execute trades
    initial_cash: Union[List[int], int] = (1000, 5000)
    transaction_fee: float = 0.025 
    bankrupt_threshold: float = 0.1  # 10% of initial balance
    slippage: float = 0.01
    seed: Optional[int] = 42
    include_base_features: bool = False # Includes base features such as timestamps and ohlc to the tensordict
    max_traj_length: Optional[int] = None
    random_start: bool = True

```

## Reward Function

Tried several reward functions but agent seems to collapse to no action at all as the transaction cost results in instant loss and negative reward. Tried sparse reward to only reward at the end of the rollout and seems to learn better.

###TODO: add reward function examples that were tested 