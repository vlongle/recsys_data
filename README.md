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

Uniform reward:
y-z:
    Empirical:
        [[2055. 2224. 2026. 2011. 1911. 1829. 2019. 2045. 1912. 1935.]
        [ 563.  585.  593.  550.  544.  553.  608.  525.  566.  546.]]
    Recurrent:
        [[1768. 2027. 1791. 1761. 1630. 1467. 1683. 1630. 1547. 1568.]
        [ 700.  786.  876.  844.  909.  952.  939.  902.  943.  877.]]
        reward = 1.4801351830363274
    Neural:
        [[1777. 2268. 2007. 2109. 1851. 1227. 1052. 1067.  974.  876.]
        [1526. 1581. 1697. 1462. 1032.  633.  606.  617.  614.  624.]]
Image:
    Recurrent:
        [[1560. 1774. 1473. 1675. 1622. 1507. 1508. 1757. 1630. 1733.]
         [ 831. 1043.  726. 1057.  772. 1046.  819. 1079.  931. 1057.]]
        reward: 1.4498666168749332
    Neural:
        [[1657. 1708. 1550. 1619. 1538. 1463. 1498. 1755. 1661. 1719.]
        [ 866. 1000.  856.  993.  890. 1024.  880. 1005.  934.  984.]]
        reward: 1.4208441208302975


Random/uniform
reduce=2
[[1266. 1370. 1303. 1388. 1215. 1162. 1297. 1289. 1319. 1256.]
 [1220. 1304. 1298. 1216. 1349. 1292. 1260. 1270. 1293. 1233.]]
reward: 1.3340656803548336
final model loss: 0.4942263066768646
time(s) : 16.617389917373657
final model accuracy: 0.8540999889373779

reduce=32
[[76. 92. 85. 87. 96. 75. 90. 79. 68. 77.]
 [66. 73. 89. 67. 94. 83. 73. 70. 72. 88.]]
reward: 0.07559878468513488
final model accuracy: 0.7044000029563904
time(s) : 16.471400260925293


Neural/uniform
[[1657. 1708. 1550. 1619. 1538. 1463. 1498. 1755. 1661. 1719.]
 [ 866. 1000.  856.  993.  890. 1024.  880. 1005.  934.  984.]]
reward: 1.4208441208302975
final model loss: 0.4611680805683136

Recurrent/uniform
[[1583. 1744. 1514. 1743. 1643. 1555. 1560. 1795. 1714. 1748.]
 [ 751. 1071.  740.  980.  735. 1016.  768. 1073.  879.  988.]]
reward: 1.4608866098523139
final model loss: 0.43689852952957153
final model accuracy: 0.873699963092804


Also has to tune these guys very carefully for it to work better than baseline...




## Reproducibility
Run 
```
CUBLAS_WORKSPACE_CONFIG=:16:8 python main_uniform.py
```
with cublas config (see https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility and https://pytorch.org/docs/stable/notes/randomness.html)

