# torchrl_alpaca_env

## Creating a Conda Environment

To create a conda environment named `alpaca_env` with Python 3.9, run the following commands:
```sh
conda create --name alpaca_env python=3.9 && 
conda activate alpaca_env
```


## Documentation

Run `mkdocs serve` to start the documentation server.

# GOAL:
The objective should be to have a set of environments, including base environments, and a simple way to construct custom environments with TorchRL for trading across different brokers. We should start with Alpaca, for example. 
Specifics for the broker, such as "get current observation" or "execute trade," should be separate packages, as these are independent functions. Ideally, we just import them and use them in the TorchRL Env. 

  
  Example Envs:
  - Start with a simple robust setup, minimal as possible that is working. then we can iterate and improve, add complexity
  - Paper trade single time frame / multi time frame
  - Live env ...
  - Offline Env versions that can read historical data from the paper trading environment.
  -  Raspberry Pi server [X]

It is essential to have independent components and to develop them stepwise, with continuous progress and rapid iteration to improve versions.

## Data can be downloaded from Market Data:
https://data.binance.vision/?prefix=data/spot/monthly/klines/BTCUSDT/1m/

# TODO:

### ALPACA
- Make env base class?
- Make multi time frame compatible
    - adapt waiting time
- find way to create combined observation -> Tensordict

- https://www.reddit.com/r/algotrading/comments/1n3zd42/live_algo_results_after_30_days_of_trading_btc_on/ INTERESTING. 1h data 5x leverage on binance. 
-  use https://neptune.ai/walkthrough for logging of training Build own dashboard for live environments.

### Offline Data Environment
Create an "server" environment that uses offline data and can be called similar to the alpaca server.

### New Vision on RL and Trading
First principle thinking. We do not need to have a sequential setup. Its more like a one step setting. Past infromation should be in the features of the market data information. Account information can be used and adapted on the fly similar to domain randomization to train on different settings/situations leading to better generalization. 

We could further use GRPO to train online on those one step data sets as it has proven to be a good algorithm for those 1 step settings. We could maybe rewrite a data sampling function and then train extremely fast iterating over those samples. 
