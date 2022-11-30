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
    PureExploitative,
)
import numpy as np

import time

import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"

"""
TODO: 
- leave_one_out scaled appropriately with the exploration factor and works now!
- each_one still doesn't work for some reasons ... probably due to the noise??? Q-values are not correct!

"""
if __name__ == "__main__":
    torch.use_deterministic_algorithms(True)
    seed_everything(0)
    start = time.time()
    num_candidates = 256
    # max_steps = 800
    max_steps = 200
    # max_steps = 200
    # max_steps = 8
    # use_img = True
    use_img = False
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
        "exploit_factor": 4.0,
        # "exploit_factor": 3.0,
        # "exploit_factor": 1.0,
    }

    estimator_cfg = {
        "reset_period": 10,
    }

    env = ImgRouterEvalEnv(cfg)

    num_tasks = 3
    num_cls = 10
    # reduce_fator = 2
    # reduce_fator = 8
    reduce_fator = 2
    num_slates = num_candidates // reduce_fator
    # estimator = EmpiricalEstimator(num_tasks, num_cls, use_img=use_img)
    # estimator = NeuralEstimator(num_tasks, num_cls, use_img=use_img)
    estimator = RecurrentNeuralEstimatorV0(
        num_tasks, num_cls, use_img=use_img, cfg=estimator_cfg)

    # estimator = DummyEstimator(num_tasks, num_cls)

    # explore = PerArmExploration(num_tasks, num_cls, num_slates)
    explore = UniformEpsilonExploration(
        num_tasks, num_cls, num_slates, explore_cfg)

    # explore = RandomRouting(num_tasks, num_cls, num_slates)

    # explore = PureExploitative(num_tasks, num_cls, num_slates, explore_cfg)

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

    cum_obs = []
    cum_action = []
    cum_rewards = []

    update_period = 10

    while not done:
        action = algo.predict(obs)
        next_obs, reward, done, info = env.step(action)

        algo.update_estimator(obs, action, info["rewards"])

        # algo.update_estimator(, cum_action,
        #                           cum_rewards, update_batch_size=update_period)
        # cum_obs.append(obs)
        # cum_action.append(action)
        # cum_rewards.append(info["rewards"])

        # # TODO: will requires some changes to the update_estimator to handle batches of data
        # # update like this...
        # if len(cum_obs) >= update_period:
        #     algo.update_estimator(cum_obs, cum_action,
        #                           cum_rewards, update_batch_size=update_period)
        #     cum_obs = []
        #     cum_action = []
        #     cum_rewards = []

        obs = next_obs
        step_rewards.append(reward)
        # pred_loss = (algo.estimator.Q - env.Q) ** 2
        # pred_losses.append(pred_loss)
        model_perfs.append(env.model.test_acc())

    print("current samples after training:")
    print(env.current_samples)
    print(env.current_samples.sum(axis=1))
    # routed_data = env.current_samples - existing_samples
    # routed_data = env.current_samples
    # print("routed data:")
    # print(routed_data)
    print("reward:", np.mean(step_rewards))
    print("final model accuracy:", model_perfs[-1])
    end = time.time()
    print("time(s) :", end - start)
