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
- exploration vs purely exploitative [DONE]
    Different seed will lead to differen artifacts!

Exploit
Seed=1
Q: [[ 0.01742403  0.01836394  0.01719892  0.01799957  0.01691584  0.01678501
   0.01683459  0.01820074  0.01792688  0.01722812]
 [-0.00595951 -0.00595951 -0.00595951 -0.00595951 -0.00595951 -0.00595951
  -0.00595951 -0.00595951 -0.00595951 -0.00595951]
 [-0.0012674  -0.00665659 -0.00329156 -0.01046107 -0.00125537 -0.0185114
  -0.00124052 -0.00165497 -0.00150916 -0.00120089]]
Q_task: [ 0.01748776 -0.00595951 -0.00470489]
[16952.    50.  8598.]
reward: 1.4187968215346336
final model accuracy: 0.7615000009536743

Seed=0

Q: [[ 0.01813336  0.01795982  0.0183684   0.01808924  0.01854646  0.01903351
   0.01832243  0.0183408   0.01848689  0.01888719]
 [-0.00415135 -0.00474086 -0.00875879 -0.00875879 -0.00511281 -0.0048538
  -0.01335531 -0.00462099 -0.00482769 -0.00464341]
 [-0.00481749 -0.00481749 -0.00481749 -0.01870573 -0.01870573 -0.00481749
  -0.00481749 -0.00481749 -0.00481749 -0.00481749]]
Q_task: [ 0.01841681 -0.00638238 -0.00759513]

[16973.  8577.    50.]
reward: 1.3606654107570648
final model accuracy: 0.7609999775886536


Exploit + explore: seed=1
Q: [[ 1.13100639e-02  1.17088602e-02  1.06703648e-02  1.13896062e-02
   1.15362278e-02  1.12665489e-02  1.07446082e-02  1.17077198e-02
   1.20111936e-02  1.11180595e-02]
 [ 2.18622793e-04  1.83515659e-05  6.63459420e-05 -1.29906748e-04
  -3.91282990e-04 -4.22152694e-04  1.84033346e-04 -1.19817042e-04
  -9.91673176e-04 -5.38825844e-04]
 [-3.51343340e-04 -7.83782585e-04 -5.13083104e-04 -2.23602082e-04
  -2.74334764e-04 -1.00785075e-03 -6.09360459e-04 -6.97850073e-04
  -7.08341364e-04 -9.94710667e-05]]
Q_task: [ 0.01134633 -0.00021063 -0.0005269 ]

[15653.  4776.  5171.]
reward: 0.8707289317250252
final model accuracy: 0.7954999804496765

seed = 0
Q: [[ 0.0129697   0.01308006  0.01251679  0.01330765  0.01233175  0.01306009
   0.01237004  0.01224659  0.01307159  0.01274496]
 [-0.00126777 -0.00200356 -0.00180125 -0.00134688 -0.00119578 -0.00162731
  -0.0011742  -0.00128394 -0.00157239 -0.0011214 ]
 [-0.00195312 -0.00154346 -0.00196072 -0.00128188 -0.00178897 -0.00178488
  -0.00149791 -0.00162342 -0.00157651 -0.00167316]]
Q_task: [ 0.01276992 -0.00143945 -0.0016684 ]
[15702.  5004.  4894.]
reward: 0.9261571273207665
final model accuracy: 0.7983999848365784

Seems like the exploit suffers from just the sampling bias on the Q-value. Q-values are actually still learned fairly well
(consistent Q_task0 >> Q_task1 ~ Q_task2), doesn't suffer from sampling bias. 

Maybe learning algorithm will?
NeuralEstimator totally freaks out!
seed=0, num_steps=200, reduce=2, NeuralEstimator.

exploit
Q: [[ 0.00106252 -0.01154559 -0.00666753 -0.00177684 -0.00114427  0.00100358
  -0.00034259 -0.00052944 -0.00196912 -0.00314587]
 [-0.00055494  0.01140538  0.01445387  0.00804806  0.01235141  0.01350413
  -0.01037217  0.01513728  0.01548397 -0.00568346]
 [ 0.01301545  0.01153415  0.01154057  0.01146931 -0.00511758  0.01083156
   0.01177508  0.01125605  0.00967828  0.01102875]]
Q_task: [-0.00250551  0.00737735  0.00970116]
current samples after training:
[[ 932.  132.  517.  864.  332.   59.  302.  303.    0.    0.]
 [   0. 1248. 1619.  228.  658. 1648.   44. 1687. 1741.   36.]
 [1635.  571. 1260. 1709.    0. 1656. 1679. 1623. 1709. 1408.]]
[ 3441.  8909. 13250.]
reward: 1.936470284461975
final model accuracy: 0.15539999306201935



exploit + explore, NN
Q: [[ 0.01283025  0.01454262  0.01216214  0.01514434  0.01481183  0.01462069
   0.0120322   0.01356644  0.01287291  0.01549344]
 [-0.00227935 -0.00420098  0.00098215 -0.00129123  0.00202055 -0.00236029
  -0.01571024  0.00110091 -0.00026717 -0.00837912]
 [-0.0033292  -0.00335167 -0.00248576  0.00072683 -0.00171792 -0.00614449
  -0.00060713 -0.00166716 -0.00620846 -0.00198073]]
Q_task: [ 0.01380769 -0.00303848 -0.00267657]
current samples after training:
[[1573. 1749. 1496. 1581. 1454. 1362. 1543. 1664. 1503. 1535.]
 [ 386.  294.  625.  319.  418.  670.  227.  602.  654.  451.]
 [ 478.  664.  463.  747.  555.  517.  912.  485.  369.  304.]]
[15460.  4646.  5494.]
reward: 1.215169062614441
final model accuracy: 0.7651000022888184




- Image vs no image [Done]
- Plot rewards, model loss and data routing [Done]
- Communication bandwidth. [Done]
- compare different evaluation method

2nd TODO:
Do some simple robotics task with curriculum training!

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



## Curriculum Learning

https://github.com/ray-project/ray/blob/master/rllib/examples/curriculum_learning.py

https://web.stanford.edu/class/aa228/reports/2020/final16.pdf

https://jmlr.org/papers/volume21/20-212/20-212.pdf


Curriculum learning via sequencing:

• Sequencing. Sequencing examines how to create a partial ordering over the set of experience samples D: that is, how to generate the edges of the curriculum graph. Most
existing work has used manually defined curricula, where a human selects the ordering
of samples or tasks. However, recently automated methods for curriculum sequencing
have begun to be explored. Each of these methods make different assumptions about
the tasks and transfer methodology used. These methods will be the primary focus
of this survey.

Replay buffer stuff:
- prioritized replay, complexity index ect to give scores to a sample.

Curriculum learning in supervised learning (Bengio et al, 2009), where training examples are presented to a learner in a specific order, rather than completely randomly


https://ronan.collobert.com/pub/2009_curriculum_icml.pdf



Supervised learning (Bengio): toy example with Gaussian "irrelevant" features. Requires the learner to be able to tell which one is "easy".

Automatic way to assign sample importance during learning progress is prioritized experience replay (RL).

TODO:
reward and Q values seem to be wrong for use_img = False for reduce_factor=8.
Slow divergence of Q-values?
- empiricalEstimator: task0_Q / task1_Q = 0.00948977 at iter=100, reduce=8
- empiricalEstimator: task0_Q / task1_Q = 4.0160075676 at iter=100, reduce=2 (a lot more discriminative!)
Probably have to tune to have more exploration if reduce = 8 for NN. Sols:
    - more exploration
    - cumulative update to stabilize the network!



>> reduce=8, empirical estimator, step=10
Q_task: [0.02138748 0.01052459 0.01316126]
current samples after training:
[118. 104.  98.]
reward: 0.49015657901763915

>> reduce=8, Neural estimator, step=10
Q_task: [0.02697795 0.02412771 0.02799636]
[ 99.  95. 126.]
reward: 0.34932780265808105


>> reduce=8, RNN estimator, step=10
Q_task: [0.02765347 0.02448561 0.02809608]
current samples after training:
[[ 8. 15.  8. 16.  9. 13.  7.  8. 10.  5.]
 [12. 10. 19.  2.  7. 10. 11.  7. 13. 18.]
 [11. 12. 16.  9. 13. 10.  8. 11. 11. 11.]]
[ 99. 109. 112.]
reward: 0.684312105178833

More exploration works!!!

reduce=8, steps=800, Neural estimator:

    explore_cfg = {
        "epsilon": 2.0,
        "min_epsilon": 0.01,
        "decay_factor": 0.95,
        "exploit_factor": 1.0,
        # "exploit_factor": 3.0,
        # "exploit_factor": 1.0,
    }


