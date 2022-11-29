import torch
from env import ImgRouterEvalEnv
from lightning_lite.utilities.seed import seed_everything
from bandit_algo import BanditAlgorithm
from preference_estimator import (
    EmpiricalEstimator,
    NeuralEstimator,
    RecurrentNeuralEstimator,
    RecurrentNeuralEstimatorV0,
    DummyEstimator,
)
from exploration_strategy import (
    PerArmExploration,
    UniformEpsilonExploration,
    RandomRouting,
)
import numpy as np

import time

import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"

"""
TODO: 
- leave_one_out scaled appropriately with the exploration factor and works now!
- each_one still doesn't work for some reasons ... probably due to the noise??? Q-values are not correct!

TODO:
- RecurrentNN works with slightly modified architecture but reset_period doesn't seem
to have an effect on the training?????
"""
if __name__ == "__main__":
    torch.use_deterministic_algorithms(True)
    seed_everything(0)
    start = time.time()
    num_candidates = 256
    max_steps = 200
    use_img = True
    cfg = {
        "num_candidates": num_candidates,
        "max_steps": max_steps,
        # "evaluate_strategy": "leave_one_out",
        # "evaluate_strategy": "each_one",
        "evaluate_strategy": "uniform",
        "use_img": use_img,
    }

    explore_cfg = {
        "epsilon": 2.0,
        "min_epsilon": 0.01,
        "decay_factor": 0.9,
    }

    estimator_cfg = {
        "reset_period": 10,
    }

    env = ImgRouterEvalEnv(cfg)

    num_tasks = 2
    num_cls = 10
    reduce_fator = 8
    num_slates = num_candidates // reduce_fator
    # estimator = EmpiricalEstimator(num_tasks, num_cls)
    # estimator = NeuralEstimator(num_tasks, num_cls, use_img=use_img)
    # estimator = RecurrentNeuralEstimator(num_tasks, num_cls, use_img=use_img)
    estimator = RecurrentNeuralEstimatorV0(
        num_tasks, num_cls, use_img=use_img, cfg=estimator_cfg)

    # estimator = DummyEstimator(num_tasks, num_cls)

    # explore = PerArmExploration(num_tasks, num_cls, num_slates)
    explore = UniformEpsilonExploration(
        num_tasks, num_cls, num_slates, explore_cfg)

    # explore = RandomRouting(num_tasks, num_cls, num_slates)
    algo = BanditAlgorithm(
        estimator,
        explore,
    )

    obs = env.reset()
    # print("initial obs:", obs)
    done = False
    step_rewards = []
    pred_losses = []
    model_perfs = [env.model.test_acc()]

    while not done:
        action = algo.predict(obs)
        next_obs, reward, done, info = env.step(action)
        algo.update_estimator(obs, action, info["rewards"])
        obs = next_obs
        step_rewards.append(reward)
        # pred_loss = (algo.estimator.Q - env.Q) ** 2
        # pred_losses.append(pred_loss)
        model_perfs.append(env.model.test_acc())

    print("current samples after training:")
    print(env.current_samples)
    # routed_data = env.current_samples - existing_samples
    # routed_data = env.current_samples
    # print("routed data:")
    # print(routed_data)
    print("reward:", np.mean(step_rewards))
    print("final model accuracy:", model_perfs[-1])
    end = time.time()
    print("time(s) :", end - start)
