from env import RouterEnv
from bandit_algo import BanditAlgorithm
from preference_estimator import (
    EmpiricalEstimator,
    NeuralEstimator,
    RecurrentNeuralEstimator,
    RecurrentNeuralEstimatorV0,
)
from exploration_strategy import (
    PerArmExploration,
    UniformEpsilonExploration,
)
from lightning_lite.utilities.seed import seed_everything
import numpy as np
from pprint import pprint


"""
TODO: 
1. visualize the loss in the prediction of the Q values of
- EmpiricalEstimator
- NeuralEstimator
- RecurrentEstimator
2. Moving from bandit to RL.

"""
if __name__ == "__main__":
    seed_everything(0)
    num_tasks = 2
    num_cls = 10
    reduce_fator = 2
    num_candidates = 64
    num_slates = num_candidates // reduce_fator
    num_samples = 24000
    # existing_samples = np.array([[0, 0],
    #                             [10, 10]])
    # target_samples = np.array([[10, 10],
    #                            [10, 10]])
    num_data_sent = num_samples // reduce_fator
    target_samples = np.ones((num_tasks, num_cls)) * (num_data_sent // num_cls)
    existing_samples = np.zeros((num_tasks, num_cls))
    existing_samples[1, :] = target_samples[1, :]
    print(target_samples)
    print(existing_samples)

    env = RouterEnv(existing_samples, target_samples,
                    cfg={"num_candidates": num_candidates,
                         "num_samples": num_samples,
                         "num_slates": num_slates,
                         "num_tasks": num_tasks,
                         "num_classes": num_cls, })

    estimator = EmpiricalEstimator(num_tasks, num_cls)
    # estimator = NeuralEstimator(num_tasks, num_cls)
    # estimator = RecurrentNeuralEstimator(num_tasks, num_cls)
    # estimator = RecurrentNeuralEstimatorV0(num_tasks, num_cls)
    explore = PerArmExploration(num_tasks, num_cls, num_slates)
    # explore = UniformEpsilonExploration(num_tasks, num_cls, num_slates)
    algo = BanditAlgorithm(
        estimator,
        explore,
    )
    obs = env.reset()
    # print("initial obs:", obs)
    done = False
    step_rewards = []
    pred_losses = []
    step = 0
    while not done:
        action = algo.predict(obs)
        next_obs, reward, done, info = env.step(action)
        algo.update_estimator(obs, action, info["rewards"])
        obs = next_obs
        step_rewards.append(reward)
        pred_loss = (algo.estimator.Q - env.Q) ** 2
        pred_losses.append(pred_loss)
        step += 1

    print("current samples after training:")
    print(env.current_samples)
    routed_data = env.current_samples - existing_samples
    print("routed data:")
    print(routed_data)
    print(np.mean(step_rewards))
    # pprint(obs)
    # pprint(env.step(np.array([0, 2])))
    # pprint(env.step(np.array([0, 1])))
