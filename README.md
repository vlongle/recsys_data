# Robotic Data Sharing as a Recommendation System
## Approach
1. Algorithm:
Q-value estimation:
    - empirical reward for each arm
    - Neural estimation of reward
    - Recurrent neural estimation
2. How to train Q?
    - Bandit: assume the reward distribution is stationary and train Q-value using 
    supervised labels of rewards.
    - RL: train using the Bellman operator to reason about long-term rewards given the
    user's internal state.

Action selection:
    - Exploit then explore: not going to work for non-stationary reward distribution
    - Epsilon-greedy: not good to be uniform over all arms
    - Epsilon per arms: similar spirit to Thompson sampling or UCB.


https://docs.ray.io/en/latest/rllib/rllib-training.html
Curriculum learning in the middle for other tasks.

## Appendix
Examples of clustering:
- Yahoo (https://www.youtube.com/watch?v=rDjCfQJ_sYY): divide users into 5 groups.