# Experience Based 1-Step Env

Taking a step back as the reward calculation and organization was too complicated to build from scratch.

It makes more sense to use the experience from a real replay buffer to build a simple one step env and then step by step add features and make it an environment that uses offline data. 


Here example why this is much easier to do and allows us to simply expand: 

```python

class OneStepTradingEnvOff(EnvBase):
    def __init__(self, tensordict: TensorDictBase, config: AlpacaTradingEnvConfig):
        self.config = config
        self.tensordict = tensordict


        self.replay_buffer = TensorDictReplayBuffer(pin_memory=False,
                                                    prefetch=4,
                                                    #split_trajs=False,
                                                    storage=LazyMemmapStorage(100000),
                                                    batch_size=1,
                                                    #shared=shared,
                                                    #sampler=SamplerWithoutReplacement(drop_last=True),
                                                    )

        self.replay_buffer.extend(self.tensordict)


    def _reset(self, tensordict: TensorDictBase, **kwargs) -> TensorDictBase:
        # here we sample from the buffer and return the current state, keep the next state as information
        # keep also the old action!
        return tensordict


    def _step(self, tensordict: TensorDictBase) -> TensorDictBase:
        # we receive the tensordict with a new action compute the reward and add the next state information
        


        return tensordict

    def _calculate_reward(self, old_portfolio_value: float, new_portfolio_value: float, action_value: float) -> float:
        # calculate the reward based on the portfolio value change
        return new_portfolio_value - old_portfolio_value

```

Like that we have the basic layout for the one step env and we can start to add features like:

- **in/out position** With fixed experience we have fixed in/out positions and thus the agent can only learn to improve the trades when in position. 
But cant improve the "get in position" part. 

- **reward** We could investigate simply different reward strategies. E.g. if the sample is not in position how do we reward the agent to get in position, if it makes sense at that time step.


All those insights and learning can then be used to build the one step environment for historic data. For example maybe we should similar to the buffer directly compute the next state features in the historic one step env. 
