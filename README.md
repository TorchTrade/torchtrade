# torchrl_alpaca_env

## Creating a Conda Environment

To create a conda environment named `alpaca_env` with Python 3.9, run the following commands:
```sh
conda create --name alpaca_env python=3.9 && 
conda activate alpaca_env
```


# TODO:

### ALPACA
- Make env base class?
- Make multi time frame compatible
    - adapt waiting time
- find way to create combined observation -> Tensordict


### Offline Data Environment
Create an "server" environment that uses offline data and can be called similar to the alpaca server.