# Alpaca Trading Environment

## TOP Priorities 

- Offline RL loop with precollected dataset for IQL / TD3-BC in /examples/offline [X]
- Make Repo a package such that we can easily import envs - collect_live/ dir with different strats. [ ]
- SimpleSequentialEnv to iterate and improve add more complex versions of it [ ]
    - To be useful in the EUREKA/ LLM hard coded strat development loop 
    - To be used in the offline loop to test the agent performance 
- OneStepEnv with offline data [ ]
    - For experimental purposes with GRPO [ ]
