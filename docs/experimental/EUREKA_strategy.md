# EUREKA strategy LLM loop

## Idea:
Similar to the [EUREKA](https://arxiv.org/abs/2310.12931) paper, we can use an LLM not to design the reward function but to design the strategy. The LLM would need to define the time frames, the features and the strategy logic. Once implemented, those strategies could be tested in offline envs or live paper trading envs. Ideally would be an offline dataset where we then rank each strategy design by its performance based on several metrics. 
Finally, the best strategy could then be deployed in a live env (paper or real).

#### Additional notes:

- [Darwin Godel Machine](https://arxiv.org/abs/2505.22954)
   We could even go more insane using Darwin Godel Machine: Open-Ended Evolution of Self-Improving Agents.
- [AlphaEvolve](https://deepmind.google/discover/blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/)
- [ShinkaEvolve](https://sakana.ai/shinka-evolve/) next generation of AlphaEvolve
