# 1-Step-Env

**Idea:** We dont need to loop over the dataset sequentially as there is no information passed between steps. The observation contains *ideally* all necessary information from the past to make a decision at the current time step.

Further, as we use tensordict and multitimeframe observations we have the situation that part of the observation is static (market data) while account information depends on balance and position information. We can precompute all static observation information and update or even sample the account information on the fly per step similar to domain randomization to make the algorithm more stable to changes in balance and position information. 


Example tensordict:

![alt text](imgs/example_td.png)


### Market Data Precomputation

### Account Information Sampling

### Reward Computation

## GRPO Integration