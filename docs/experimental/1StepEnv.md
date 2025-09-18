# 1-Step-Env

**Idea:** We dont need to loop over the dataset sequentially as there is no information passed between steps. The observation contains *ideally* all necessary information from the past to make a decision at the current time step.

Further, as we use tensordict and multitimeframe observations we have the situation that part of the observation is static (market data) while account information depends on balance and position information. We can precompute all static observation information and update or even sample the account information on the fly per step similar to domain randomization to make the algorithm more stable to changes in balance and position information. 

At each step the agent is given a market observation that can be randomly sampled from the dataset (or sequentially) and a partly random account information. Such that the agent at each time step is confronted with a new situation. Which might be at 50% of the time the agent is not in position and at 50% of the time the agent is in position. If the agent is in position it needs to decide if based on the current market observation it should sell or hold. If the agent is not in position it needs to decide if based on the current market observation it should buy or hold.

**Reward in position:**
Reward for in position is computed if the agent makes the decision to sell. Then we would compute the reward based on the difference between the buy price and the sell price at the next timestamp. Some noise can be added to the next timestamp to make the environment more realistic and simulate market volatility and slippage.

    Buy price could be selected from the last x time steps: randomly, min, mean,...

**Reward out of position:**
Reward for out of position is computed if the agent makes the decision to buy. Then we would compute the reward based on the difference between the buy price and the sell price at the next timestamp or some time step in the future. Some noise can be added to the next timestamp to make the environment more realistic and simulate market volatility and slippage.

    Sell price could be selected from the next x time steps: randomly, min, mean,...

Example tensordict:

![alt text](imgs/example_td.png)


### Market Data Precomputation

Sampler is initialized and filters dataset depending on the timeframes and window sizes, exactly as in the live environment:

```python
class MarketDataObservationSampler():
    def __init__(
        self,
        df: pd.DataFrame,
        time_frames: Union[List[TimeFrame], TimeFrame] = TimeFrame(1, TimeFrameUnit.Minute),
        window_sizes: Union[List[int], int] = 10,
        execute_on: TimeFrame = TimeFrame(1, TimeFrameUnit.Minute),
        feature_processing_fn: Optional[Callable] = None,
        features_start_with: str = "features_"
    ):
```

#### Get Observation


We have currently 2 different ways of sample observations:

1. Random Sampling w/wo replacement
2. Sequential Sampling

```python
    def get_random_observation(self, without_replacement: bool = False)->Tuple[Dict[str, pd.DataFrame], pd.Timestamp]:
        """Get a random observation from the dataset.
        If without_replacement is True, the timestamp is removed from the list of unseen timestamps.
        """
        timestamp = self.get_random_timestamp(without_replacement)
        return self.get_observation(timestamp), timestamp

    def get_sequential_observation(self)->Tuple[Dict[str, pd.DataFrame], pd.Timestamp]:
        """Get the next observation in the dataset.
        The timestamp is removed from the list of unseen timestamps.
        """
        timestamp = self.unseen_timestamps.pop(0)
        return self.get_observation(timestamp), timestamp

```

The sampler returns the observation and the depending timestamp when sampling and observation.

Max steps for the sampler can be retrieved with:

```python
    def get_max_steps(self)->int:
        return self.max_steps
```

Once we have sampled all observations the sampler is reset and can be used again:

```python
    def reset(self)->None:
        """Reset the observation sampler."""
        self.unseen_timestamps = list(self.exec_times)
```

Example Observation output from the sampler:

![alt text](imgs/sampler_example_obs.png)


#### Next Observation

**Retrieve the next observation from the sampler.**
    We return the time stamp for get_random_observation and get_sequential_observation. This time stamp can be used to get the next observation:
    ```python
    from pandas import Timedelta
    obs, timestamp = sampler.get_random_observation()
    next_timestamp = timestamp + Timedelta(minutes=5) # if we execute on 5 min timeframe
    next_obs = sampler.get_observation(next_timestamp)
    ``` 

We have a tutorial notebook that shows how to use the sampler: [Tutorial Notebook](../tutorials/1StepEnv.ipynb)


#### Base Features

For some calculations like reward you might need the base open, high, low, close, volume, information. This can be retrieved from the sampler with:

```python
    def get_base_features(self, timestamp: pd.Timestamp)->pd.DataFrame:
        """Get the base features from the dataset at the given timestamp."""
        return self.execute_base_features.loc[timestamp]
```

    

### Account Information Sampling


    # TODO: implement.

    Currently the account information is [cash, portfolio_value, position_size]. Should we add buy price? (sell if short) -> getting in position price.


#### buy entry price selection Example 

```python
def sample_entry_price(t, price_series, volume_series=None,
                       method='uniform', k=20, alpha=0.3, slip_params=None):
    # price_series: array indexed by time with fields 'open','mid', ...
    # returns P_entry_exec, entry_idx
    window_indices = range(max(0, t-k), t)  # strictly before t
    opens = [price_series[i].open for i in window_indices]

    if method == 'uniform':
        entry_idx = random.choice(window_indices)
        P_base = price_series[entry_idx].open

    elif method == 'min':
        entry_idx = window_indices[np.argmin(opens)]
        P_base = min(opens)

    elif method == 'mean':
        entry_idx = t-1  # or choose representative index
        P_base = sum(opens)/len(opens)

    elif method == 'ewma':
        weights = [(1-alpha)**(len(opens)-1-i) for i in range(len(opens))]
        weights = [w/sum(weights) for w in weights]
        P_base = sum(w*p for w,p in zip(weights, opens))
        entry_idx = t-1  # EWMA considered 'recent'

    elif method == 'vwap':
        assert volume_series is not None
        vols = [volume_series[i] for i in window_indices]
        P_base = sum(p*v for p,v in zip(opens, vols)) / sum(vols)
        entry_idx = t-1

    elif method == 'empirical':
        # sample a historical hold time d, then set entry_idx = t-d if valid
        d = sample_hold_time_from_empirical()
        entry_idx = max(0, t-d)
        P_base = price_series[entry_idx].open

    # add execution slippage
    slip = sample_slippage(slip_params)  # e.g. Normal(mu, sigma), clipped
    P_entry_exec = P_base * (1 + abs(slip))  # long entry pays ask

    return P_entry_exec, entry_idx
```

### Reward Computation



## GRPO Integration