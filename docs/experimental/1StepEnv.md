# 1-Step-Env

**Idea:** We dont need to loop over the dataset sequentially as there is no information passed between steps. The observation contains *ideally* all necessary information from the past to make a decision at the current time step.

Further, as we use tensordict and multitimeframe observations we have the situation that part of the observation is static (market data) while account information depends on balance and position information. We can precompute all static observation information and update or even sample the account information on the fly per step similar to domain randomization to make the algorithm more stable to changes in balance and position information. 


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
        feature_processing_fn: Optional[Callable] = None
    ):
```

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



### TODO:
- make feature processing function work
- add next observation. Ideally we want to receive current and next observation. maybe add include next obs as flag. 

### Account Information Sampling

### Reward Computation

## GRPO Integration