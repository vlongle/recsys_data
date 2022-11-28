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

### MNIST
y,z routing works pretty well.
TODO: use the image directly instead of y, z for the neural and recurrent router.

https://docs.ray.io/en/latest/rllib/rllib-training.html
Curriculum learning in the middle for other tasks.

## Appendix
Examples of clustering:
- Yahoo (https://www.youtube.com/watch?v=rDjCfQJ_sYY): divide users into 5 groups.

More non-stationary env with bandits:
https://www.yoanrussac.com/en/talk/talk1-ens/intro_linear_bandits.pdf

On the uniform reward, neural estimator is extremely turbulent.

## Active Learning
Seems like Active learning is particularly useful in RL setting where students "ask" for
help when it got stuck. What's the correct behavior / trajectory for behavior cloning in the state space where it is not sure what to do.

Learning how to Active Learn: A Deep Reinforcement Learning Approach



## Test Improvement
- compute test performance improvement per point. (going to be slow, very noisy as well.)
- sample and compute the average (not super principle)
- leave one out (Less noisy but also very slow.)

Problems:
    Quite noisy labels. Might still work though I don't know...


Diff is also a bit noisy but it is probably better than any other methods

