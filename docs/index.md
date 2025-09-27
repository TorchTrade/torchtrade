# Alpaca Trading Environment

## TOP Priorities 

- Clean Examples:
    - make online examples clean: add more logging for eval like number of actions (buy/sell) take during eval,look for other interesting metrics to log (sharp, etc)
    - make model creation more clean and simple: encoder blocks should be created in a loop depending on how many market_data informations are given
    - add eval_env to offline example. 
    - add PPO example 
    - make it use compile? faster training?

- Package:
    - Make Repo a package such that we can easily import envs [X]
    - live dir in which we have simple live scripts: eg. random_live.py iql_live.py... [ ]

- OneStepEnv with offline data [ ]
    - For experimental purposes with GRPO [ ]

- How can we sweep over hyperparameters and other elements like market_data elements?
    - what is a good strategy?
    
- How can we scale? 




LLM-Agent-Lab Repo
- create agent lab repo in which hard coded strats can be developed.
- look for experimental docs...