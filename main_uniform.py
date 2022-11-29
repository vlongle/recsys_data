from env import ImgRouterEvalEnv
from lightning_lite.utilities.seed import seed_everything
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
import numpy as np

import time

"""
TODO: 
- leave_one_out scaled appropriately with the exploration factor and works now!
- each_one still doesn't work for some reasons ... probably due to the noise??? Q-values are not correct!

"""
if __name__ == "__main__":
    seed_everything(0)

    start = time.time()
    num_candidates = 256
    max_steps = 200
    cfg = {
        "num_candidates": num_candidates,
        "max_steps": max_steps,
        "evaluate_strategy": "leave_one_out",
        # "evaluate_strategy": "each_one",
        # "evaluate_strategy": "uniform",
    }

    explore_cfg = {
        "epsilon": 2.0,
        "min_epsilon": 0.01,
        "decay_factor": 0.9,
    }
    env = ImgRouterEvalEnv(cfg)

    num_tasks = 2
    num_cls = 10
    reduce_fator = 2
    num_slates = num_candidates // reduce_fator
    # estimator = EmpiricalEstimator(num_tasks, num_cls)
    estimator = NeuralEstimator(num_tasks, num_cls)
    # estimator = RecurrentNeuralEstimatorV0(num_tasks, num_cls)
    # explore = PerArmExploration(num_tasks, num_cls, num_slates)
    explore = UniformEpsilonExploration(
        num_tasks, num_cls, num_slates, explore_cfg)
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
        # pred_loss = (algo.estimator.Q - env.Q) ** 2
        # pred_losses.append(pred_loss)
        step += 1

    print("current samples after training:")
    print(env.current_samples)
    # routed_data = env.current_samples - existing_samples
    # routed_data = env.current_samples
    # print("routed data:")
    # print(routed_data)
    print("reward:", np.mean(step_rewards))
    end = time.time()
    print("time(s) :", end - start)
