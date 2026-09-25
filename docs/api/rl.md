# Reinforcement Learning API Reference

The `chokkhu.models.rl` package implements value-based, policy gradient, actor-critic, and sequence-modeling offline reinforcement learning algorithms in pure NumPy.

---

## 1. Value-Based Algorithms

### Deep Q-Network Family
- `DQN`: Q-learning with target networks and experience replay.
- `DoubleDQN`: Decouples action selection from action evaluation to eliminate Q-value overestimation bias:
  $$Y_t^{\text{DoubleQ}} = R_{t+1} + \gamma Q(S_{t+1}, \arg\max_a Q(S_{t+1}, a; \theta_t); \theta_t^-)$$
- `DuelingDQN`: Decomposes state-action values into state values and action advantages: $Q(s,a) = V(s) + \left( A(s,a) - \frac{1}{|\mathcal{A}|}\sum_{a'} A(s,a') \right)$.
- `PrioritizedReplayBuffer`: Samples transitions proportionally to temporal difference error: $P(i) = \frac{p_i^\alpha}{\sum_k p_k^\alpha}$.

---

## 2. Policy Gradient & Actor-Critic Algorithms

### Proximal Policy Optimization (`PPO`)
Clips surrogate objective ratio to prevent destabilizing large policy updates:

$$L^{\text{CLIP}}(\theta) = \hat{\mathbb{E}}_t \left[ \min\left( r_t(\theta) \hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t \right) \right], \quad r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)}$$

Equipped with Generalized Advantage Estimation (GAE-$\lambda$):

$$\hat{A}_t^{\text{GAE}} = \sum_{l=0}^\infty (\gamma \lambda)^l \delta_{t+l}^V, \quad \delta_t^V = R_{t+1} + \gamma V(S_{t+1}) - V(S_t)$$

### Soft Actor-Critic (`SAC`)
Maximum entropy off-policy actor-critic optimizing reward plus policy entropy $\mathcal{H}(\pi(\cdot|s_t))$.

---

## 3. Offline RL & Transformers

### Decision Transformer (`DecisionTransformer`)
Frames reinforcement learning as conditional autoregressive sequence modeling over trajectories $\tau = (\hat{R}_1, s_1, a_1, \hat{R}_2, s_2, a_2, \dots)$ using causal self-attention.

## Example

```python
import numpy as np
from chokkhu.models.rl.q_learning import QLearningAgent

agent = QLearningAgent(n_states=10, n_actions=4)
state = 0
action = agent.select_action(state)
agent.update(state, action, reward=1.0, next_state=1, done=False)
print("Q-values updated for state", state)
```
