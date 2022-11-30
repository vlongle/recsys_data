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



TODO:
1. modify the uniform.ipynb to make it reproducible with main_uniform.py
2. Plot the number of data points sent per tasks!
3. Final paper: 
- exploration vs purely exploitative
- Image vs no image 
- Plot rewards, model loss and data routing [Done]
- Communication bandwidth. [Done]
- compare different evaluation method


TODO: for large reduce factor, learning estimators are pretty bad. 
DEBUG: look at the Q-value for the img=False. Is the Q correct?
1. Maybe too noisy --> need to update the neural models only on batch.
2. Maybe exploration vs exploit issue. Need to tune explore_config?



Exploit factor =1, Empirical, uniform reward.
    array([12452.,  5173.,  4871.])
    EmpiricalEstimator 0.681027113518212


Exploit factor = 3, Empirical, uniform reward
    Data: array([17630.,  2516.,  2350.])
    EmpiricalEstimator 0.7263356351494704



With exploit factor = 3, NN still sucks probably due to noisy update...

update_period = 1

current samples after training:
[[ 265.  157.   88.   57.   49.   32.   35.   46.   40.   61.]
 [5043. 4879.  347.   76.   52.   36.   31.   44.   57.  120.]
 [3767. 4848. 1617.  197.   70.   46.   41.   98.  158.  139.]]
reward: 14.159670521490604
final model accuracy: 0.18029999732971191
time(s) : 77.77452874183655





sizes = [4, 2, 6]
action = [[0], [0], [0]]


TODO: debug!!!!!
NeuralNetEstimator straight up does NOT WORK ANY MORE, WTF??????????????????????


update_period code might be buggy...



TODO: move actions out of update function in preference_estimator.


TODO: use_img = False is BROKEN for neural estimator and recurrent estimator now...

rewards and stuff are all WRONG!!!
bs



Exploit vs explore + exploit

Explore + exploit: exploit_factor = 4, NN, reduce_factor=2

[1323. 1578. 1573. 1714. 1566. 1512. 1592. 1629. 1444. 1368.]
 [1527. 1532. 1492. 1324. 1019.  627.  342.  259.  161.  148.]
 [ 318.  264.  205.  209.  167.  151.  143.  153.  132.  128.]]
[15299.  8431.  1870.]
reward: 0.8757922601699829
final model accuracy: 0.8307999968528748

exploit: 



NOTE: use_img = False might still be buggy since [1, 0] classes still have a large routing for some reasons...

[1, 0], [1, 1], [1, 2], [1, 3], [1, 4]


batch_y is still wrong...


Somehow NN is biased towards task 1 now with use_img = False. Some sort of weird embedding with the tasks probably.
Move to nn.embedding to solve this problem!

New NN: explore + exploit
    [[1573. 1749. 1496. 1581. 1454. 1362. 1543. 1664. 1503. 1535.]
    [ 386.  294.  625.  319.  418.  670.  227.  602.  654.  451.]
    [ 478.  664.  463.  747.  555.  517.  912.  485.  369.  304.]]
    [15460.  4646.  5494.]
    reward: 1.215169062614441

New NN: only exploit
[[ 932.  132.  517.  864.  332.   59.  302.  303.    0.    0.]
 [   0. 1248. 1619.  228.  658. 1648.   44. 1687. 1741.   36.]
 [1635.  571. 1260. 1709.    0. 1656. 1679. 1623. 1709. 1408.]]
[ 3441.  8909. 13250.]
reward: 1.936470284461975
final model accuracy: 0.15539999306201935
It's bad! which is good for us! but it doesn't follow the pattern I was expecting...
Needs to re-check this...

EmpiricalEstimator:
current samples after training:
[[1534. 1778. 1594. 1598. 1552. 1396. 1503. 1660. 1540. 1547.]
 [ 534.  472.  475.  470.  504.  518.  544.  504.  432.  551.]
 [ 543.  540.  397.  478.  456.  465.  519.  464.  532.  500.]]
[15702.  5004.  4894.]
reward: 0.9261571273207665
final model accuracy: 0.7983999848365784


EmpiricalEstimator: only exploit
urrent samples after training:
[[1641. 1906. 1713. 1727. 1693. 1523. 1624. 1807. 1658. 1681.]
 [1732. 1570.   10.   10.  315. 1571.   11. 1351.  459. 1548.]
 [   3.    2.    5.    6.    6.    4.    5.    4.    9.    6.]]
[16973.  8577.    50.]
reward: 1.3606654107570648
final model accuracy: 0.7609999775886536
time(s) : 18.928626775741577
TODO: look at the trace to verify why pure exploitation is bad.


Random:
[[788. 943. 863. 859. 853. 754. 818. 908. 872. 843.]
 [875. 838. 834. 837. 897. 836. 845. 836. 840. 828.]
 [899. 850. 889. 845. 843. 860. 847. 873. 896. 831.]]
[8501. 8466. 8633.]
reward: 0.4124806725978851
final model accuracy: 0.6301999688148499


RNN (before the embedding fix)

current samples after training:
[[1462. 1669. 1431. 1506. 1435. 1418. 1524. 1621. 1479. 1537.]
 [1492. 1408. 1282. 1044.  846.  679.  549.  490.  415.  349.]
 [ 242.  229.  187.  216.  171.  203.  207.  166.  180.  163.]]
[15082.  8554.  1964.]
reward: 0.8803443545103073
final model accuracy: 0.8504999876022339

RNN (after the fix!!)

current samples after training:
[[1540. 1716. 1602. 1588. 1549. 1439. 1590. 1650. 1502. 1503.]
 [ 369.  168.  487.  685.  379.  434.  692.  147.  519.  415.]
 [ 210.  711.  730.  552.  438.  697.  414.  656.  492.  726.]]
[15679.  4295.  5626.]
reward: 0.8368818444013596
final model accuracy: 0.7346000075340271

