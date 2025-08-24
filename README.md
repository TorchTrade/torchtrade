# torchrl_alpaca_env

## Creating a Conda Environment

To create a conda environment named `alpaca_env` with Python 3.9, run the following commands:
```sh
conda create --name alpaca_env python=3.9 && 
conda activate alpaca_env
```


# GOAL:
The objective should be to have a set of environments, including base environments, and a simple way to construct custom environments with TorchRL for trading across different brokers. We should start with Alpaca, for example. 
Specifics for the broker, such as "get current observation" or "execute trade," should be separate packages, as these are independent functions. Ideally, we just import them and use them in the TorchRL Env. 

  
  Example Envs:
    - Paper trade single time frame / multi time frame
    - Live env ...
    - Offline Env versions that can read historical data from the paper trading environment.
    ...

It is essential to have independent components and to develop them stepwise, with continuous progress and rapid iteration to improve versions.



# TODO:

### ALPACA
- Make env base class?
- Make multi time frame compatible
    - adapt waiting time
- find way to create combined observation -> Tensordict


### Offline Data Environment
Create an "server" environment that uses offline data and can be called similar to the alpaca server.
