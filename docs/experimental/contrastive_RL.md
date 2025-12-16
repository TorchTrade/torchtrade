# Contrastive RL for Trading

Contrastive learning is interesting because it replaces fragile, hand-designed rewards with a self-supervised signal that teaches an agent which states and actions are meaningfully related. Instead of predicting exact values, the model learns relative structure (“this goal is more reachable than others”), which gives dense gradients, stable learning under distribution shift, and scales naturally with model depth. In RL, this turns control into representation learning + inference, unlocking long-horizon behavior without explicit rewards.

## Paper:
- [Contrastive Learning as Goal-Conditioned Reinforcement Learning](https://arxiv.org/pdf/2206.07568)
- [1000 Layer Networks for Self-Supervised RL: Scaling Depth Can Enable New Goal-Reaching Capabilities](https://arxiv.org/abs/2503.14858)