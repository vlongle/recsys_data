from utils import get_batch_tasks_cls
from abc import abstractmethod
import numpy as np
from typing import (
    Optional,
    Dict,
)


class ExplorationStrategy:
    @abstractmethod
    def get_action(self, observations: np.ndarray, Q_values: np.ndarray):
        pass

    @abstractmethod
    def update(self, observations: np.ndarray, actions: np.ndarray):
        pass


class PerArmExploration(ExplorationStrategy):
    def __init__(self, num_tasks: int, num_cls: int, num_slates: int):
        self.num_tasks = num_tasks
        self.num_cls = num_cls
        self.num_slates = num_slates
        self.num_chosens = np.zeros((num_tasks, num_cls))
        self.num_data_sent = num_tasks * num_cls
        self.epsilon = 2.0
        self.min_epislon = 0.01
        self.decay_factor = 0.9

    def update(self, observations: np.ndarray, actions: np.ndarray):
        """
        Args:
            observations: (batch_size, 2) array of (task, class) tuples.
            actions: (num_slates, ) array of actions.
        """
        batch_tasks, batch_cls = get_batch_tasks_cls(observations, actions)
        for task in range(self.num_tasks):
            for c in range(self.num_cls):
                self.num_chosens[task, c] += np.sum(
                    (batch_tasks == task) & (batch_cls == c))
        self.num_data_sent += actions.shape[0]
        self.epsilon = max(self.min_epislon,
                           self.epsilon * self.decay_factor)

    def get_action(self, observations: np.ndarray, Q_values: np.ndarray):
        """
        Args:
            observations: (batch_size, 2) array of (task, class) tuples.
            Q_values: (batch_size,) array of Q values.
        Returns:
            action: (num_slates,) array of actions.
        """
        tasks, cls = observations[:, 0], observations[:, 1]
        explore_factor = (np.log(self.num_data_sent) /
                          np.maximum(self.num_chosens[tasks, cls], 1)) ** (0.5) * self.epsilon
        # replace nan by 1
        weights = explore_factor + Q_values
        # clamp weights to >= 0
        # HACK: should probably takes exp(weights) and then normalize
        # since clamping to 0 means that negative rewards / Qs will never be picked
        # again, which is appropriate in this oracle preference but might not be true
        # in general (e.g., test improvement)
        # weights = np.maximum(weights, 0)
        weights = np.exp(weights / self.epsilon)
        probs = weights / np.sum(weights)
        # sample without replacement if there's enough non-zero weights for num_slates
        # otherwise, send all non-zero weights
        if np.sum(weights > 0) < self.num_slates:
            action = np.nonzero(weights)[0]
            # if action is empty, then all weights are 0, so just a random batch
            # (equally likely to be any batch)
            if action.shape[0] == 0:
                action = np.random.choice(
                    np.arange(observations.shape[0]), self.num_slates)
        else:
            action = np.random.choice(
                observations.shape[0], self.num_slates, p=probs, replace=False)
        return action


class UniformEpsilonExploration(ExplorationStrategy):
    def __init__(self, num_tasks, num_cls, num_slates, cfg: Optional[Dict] = {}) -> None:
        self.num_slates = num_slates
        self.epsilon = cfg.get("epsilon", 2.0)
        self.min_epislon = cfg.get("min_epislon", 0.01)
        self.decay_factor = cfg.get("decay_factor", 0.9)
        self.step = 0

    def get_action(self, observations: np.ndarray, Q_values: np.ndarray):
        explore_factor = self.epsilon
        weights = explore_factor + Q_values
        # replace nan by 1
        # clamp weights to >= 0
        # HACK: should probably takes exp(weights) and then normalize
        # since clamping to 0 means that negative rewards / Qs will never be picked
        # again, which is appropriate in this oracle preference but might not be true
        # in general (e.g., test improvement)
        # weights = np.maximum(weights, 0)
        weights = np.exp(weights / self.epsilon)
        probs = weights / np.sum(weights)
        # sample without replacement if there's enough non-zero weights for num_slates
        # otherwise, send all non-zero weights
        if np.sum(weights > 0) < self.num_slates:
            print("sending all non-zero weights")
            action = np.nonzero(weights)[0]
            print()
            # if action is empty, then all weights are 0, so just a random batch
            # (equally likely to be any batch)
            if action.shape[0] == 0:
                action = np.random.choice(
                    np.arange(observations.shape[0]), self.num_slates)
        else:
            action = np.random.choice(
                observations.shape[0], self.num_slates, p=probs, replace=False)
        return action

    def update(self, observations: np.ndarray, actions: np.ndarray):
        print("step {} epsilon {}".format(self.step, self.epsilon))
        self.epsilon = max(self.min_epislon,
                           self.epsilon * self.decay_factor)
        self.step += 1


class RandomRouting(ExplorationStrategy):
    def __init__(self, num_tasks, num_cls, num_slates, cfg={}):
        self.num_slates = num_slates

    def get_action(self, observations: np.ndarray, Q_values: np.ndarray):
        return np.random.choice(
            observations.shape[0], self.num_slates, replace=False)

    def update(self, observations: np.ndarray, actions: np.ndarray):
        pass
