# Plans for Dezember 25

**Big Picture:**
- [X] 43% speed improvement for offline envs thanks to marketdataobservation sampler.
      Summary:
        All performance improvements for issue #27 have been implemented and the PR is ready for review:

        Key optimizations:
        - deque.popleft() instead of list.pop(0) - O(1) vs O(n)
        - tolist() instead of 5x .item() calls - 14x faster
        - Cached get_base_features() to eliminate redundant calls per step
        - Fixed UnboundLocalError bug in _rollout()

        Results:
        - ~2,907 steps/s → ~8,269 steps/s (~43% improvement)
        - All 303 tests pass
        - Training example runs successfully

- [X] Add Binance env
    - [X] Binance Live
    - [X] Short-Long Environment
    - [X] One Step Short-Long Env version

- [ ] OANDA Env 
    - [ ] 

- [X] Gather Bigger Dataset

- [X] Find ways to utilize Claude skills / commands
    - [X] TorchRL Algorithm Implementation Expert
    - [ ] Automate experimentation


**Smaller things:**
- [ ] Fix open issues regarding data leakage - verify if really true
- [ ] Find a way to run sweeps effectively 
- [ ] Look for ways to 

**Possible things:**

- VLM Actor with human actor dashboard
- Contrastive RL 