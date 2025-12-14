# SeqLongOnlySLTPEnv
Environment for sequential long-only trading where the agent decides what `stop loss` and `take profit` level to use.
Used to train agents for the alpaca crpyto trading as there is long only actions.

> NOTE: Currently we continue trading even though a position is opened. The agent keeps taking actions while he cant do anything. We should not do so but "spin" forward in time once the sell action is triggered which forms the next state (?). How should we actually formulize the next state then. But reward calculation is easier as we can take the reward directly from initial to exit. Also the agent will not be confused by all the other actions it can take until selling is triggered!  --> similar to One-Step Env. 

> Maybe this update opens up the option for GRPO style training? As we could "simulate" the buying for multiple options across several rollouts

## Observation Spec
Flexible observation spec that can be defined by the user via the time_frames, window_sizes and feature_preprocessing_fn.
Utilizes the `MarketDataObservationSampler` to sample market data single or multi time frame. 


## Action Spec
The action space for the actions the agent can take is flexible and depends on the stoploss_levels and takeprofit_levels that the user defines in the `SeqLongOnlySLTPEnvConfig`. Action space is a combinatorial space of all possible stop loss and take profit levels + 1 `no-action` option.

## Config 

```python

class SeqLongOnlySLTPEnvConfig:
    symbol: str = "BTC/USD"
    time_frames: Union[List[TimeFrame], TimeFrame] = TimeFrame(1, TimeFrameUnit.Minute)
    window_sizes: Union[List[int], int] = 10
    execute_on: TimeFrame = TimeFrame(1, TimeFrameUnit.Minute) # On which timeframe to execute trades
    initial_cash: Union[List[int], int] = (1000, 5000)
    transaction_fee: float = 0.025
    stoploss_levels: Union[List[float], float] = (-0.025, -0.05, -0.1)
    takeprofit_levels: Union[List[float], float] = (0.05, 0.1, 0.2)
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