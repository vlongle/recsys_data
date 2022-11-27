
from lightning_lite.utilities.seed import seed_everything
import gym
from typing import (
    Optional,
    Tuple,
    Dict)

import numpy as np
from utils import get_batch_tasks_cls


class RouterEnv(gym.Env):
    """
    Observation: (task, class) tuple of size num_candidates at each time step.
    Action: a sequence of size (num_slates, ) choosing the datapoints to route.
    """

    def __init__(self, existing_samples: np.ndarray, target_samples: np.ndarray, cfg: Optional[Dict] = {}) -> None:
        super().__init__()
        self.existing_samples = existing_samples
        self.target_samples = target_samples
        self.num_tasks = cfg.get("num_tasks", 2)
        self.num_classes = cfg.get("num_classes", 2)
        self.num_candidates = cfg.get("num_candidates", 10)
        self.num_slates = cfg.get("num_slates", 2)
        self.num_samples = cfg.get("num_samples", 100)
        self.max_env_steps = self.num_samples // self.num_candidates

    def _generate_data(self) -> np.ndarray:
        """
        Generate a random data stream.
        Returns:
            data_stream: (num_samples, 2) array of (task, class) tuples.
        """
        rand_tasks = np.random.randint(self.num_tasks, size=self.num_samples)
        rand_classes = np.random.randint(
            self.num_classes, size=self.num_samples)
        return np.stack([rand_tasks, rand_classes], axis=1)

    def _get_obs(self) -> np.ndarray:
        """
        Get the current batch of data.
        Returns:
            obs: (num_candidates, 2) array of (task, class) tuples.
        """
        return self.data_stream[self.env_step * self.num_candidates:
                                (self.env_step + 1) * self.num_candidates]

    def reset(self) -> None:
        self.data_stream = self._generate_data()
        self.env_step = 0
        self.current_samples = self.existing_samples.copy()
        return self._get_obs()

    def _get_rewards(self, routed_batch: np.ndarray) -> np.ndarray:
        """
        Compute the reward for the given action.
        Args:
            routed_batch: (num_slates, 2) array of (task, class) tuples.
        Returns:
            reward: float reward.
        """
        assert routed_batch.shape[
            0] <= self.num_slates, "Action must be of size (<=num_slates, )"
        assert routed_batch.shape == (
            routed_batch.shape[0], 2), "Invalid routed batch shape"
        # parse to task and class
        batch_tasks = routed_batch[:, 0]
        batch_cls = routed_batch[:, 1]
        return (self.target_samples[batch_tasks, batch_cls] - self.current_samples[batch_tasks, batch_cls]) / self.target_samples[batch_tasks, batch_cls]

    @property
    def Q(self) -> np.ndarray:
        return (self.target_samples - self.current_samples) / self.target_samples

    def step(self, action: np.ndarray) -> Tuple:
        assert action.shape[0] <= self.num_slates, "Action must be of size (<=num_slates, )"
        assert action.shape == (
            action.shape[0], ), "Action must be of size (num_slates, )"
        current_batch = self._get_obs()
        routed_batch = current_batch[action]
        rewards = self._get_rewards(routed_batch)
        # update the states
        self.env_step += 1
        batch_tasks, batch_cls = get_batch_tasks_cls(current_batch, action)
        for task in range(self.num_tasks):
            for cls in range(self.num_classes):
                self.current_samples[task,
                                     cls] += np.sum((batch_tasks == task) & (batch_cls == cls))
        return self._get_obs(), np.sum(rewards), self.env_step >= self.max_env_steps, {"rewards": rewards}


if __name__ == "__main__":
    seed_everything(1)
    existing_samples = np.array([[0, 0],
                                [10, 10]])
    target_samples = np.array([[10, 10],
                               [10, 10]])

    env = RouterEnv(existing_samples, target_samples,
                    cfg={"num_candidates": 5})
    env.reset()
    obs = env._get_obs()
    print(obs)
    env.step(np.array([0, 3]))
