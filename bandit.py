from lightning_lite.utilities.seed import seed_everything
import numpy as np
from utils import get_batch_tasks_cls


class EmpiricalEstimator:
    def __init__(self, num_tasks: int, num_cls: int):
        self.num_tasks = num_tasks
        self.num_cls = num_cls
        self.cum_rewards = np.zeros((num_tasks, num_cls))
        self.num_chosens = np.ones((num_tasks, num_cls))

    @property
    def Q(self) -> np.ndarray:
        """
        Compute the Q values.
        Returns:
            Q: (num_tasks, num_cls) array of Q values.
        """
        return self.cum_rewards / self.num_chosens

    def update(self, observations: np.ndarray, actions: np.ndarray, rewards: np.ndarray):
        """
        Update the Q values.
        Args:
            observations: (batch_size, 2) array of (task, class) tuples.
            actions: (batch_size, ) array of actions.
            rewards: (batch_size, ) array of rewards.
        """
        assert rewards.shape == (
            actions.shape[0], ), "Rewards and actions must have the same shape."
        batch_tasks, batch_cls = get_batch_tasks_cls(observations, actions)
        for task in range(self.num_tasks):
            for c in range(self.num_cls):
                self.cum_rewards[task, c] += np.sum(
                    ((batch_tasks == task) & (batch_cls == c)) * rewards)
                self.num_chosens[task, c] += np.sum(
                    (batch_tasks == task) & (batch_cls == c))

    def get_Q(self, observations: np.ndarray):
        """
        Args:
            observations: (batch_size, 2) array of (task, class) tuples.
        """
        batch_tasks, batch_cls = observations[:, 0], observations[:, 1]
        return self.Q[batch_tasks, batch_cls]


class PerArmExploration:
    def __init__(self, num_tasks: int, num_cls: int, num_slates: int):
        self.num_tasks = num_tasks
        self.num_cls = num_cls
        self.num_slates = num_slates
        self.num_chosens = np.ones((num_tasks, num_cls))
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
        # NOTE: this explore_factor needs to be tuned otherwise, we might hit a
        # lot of unnecessary tasks!
        # explore_factor = np.log(self.num_data_sent) / \
        #     self.num_chosens[tasks, cls]
        explore_factor = (np.log(self.num_data_sent) /
                          self.num_chosens[tasks, cls]) * self.epsilon
        # replace nan by 1
        weights = explore_factor + Q_values
        # clamp weights to >= 0
        # HACK: should probably takes exp(weights) and then normalize
        # since clamping to 0 means that negative rewards / Qs will never be picked
        # again, which is appropriate in this oracle preference but might not be true
        # in general (e.g., test improvement)
        weights = np.maximum(weights, 0)
        probs = weights / np.sum(weights)
        # sample without replacement if there's enough non-zero weights for num_slates
        # otherwise, send all non-zero weights
        if np.sum(weights > 0) < self.num_slates:
            action = np.nonzero(weights)[0]
        else:
            action = np.random.choice(
                observations.shape[0], self.num_slates, p=probs, replace=False)
        return action


class Algorithm:
    def __init__(self, estimator, exploration_strategy):
        self.estimator = estimator
        self.exploration_strategy = exploration_strategy

    def predict(self, obs: np.ndarray):
        """
        obs: (batch_size, num_features) array of observations.
        """
        Q_values = self.estimator.get_Q(obs)
        action = self.exploration_strategy.get_action(obs, Q_values)
        self.exploration_strategy.update(obs, action)
        return action

    def update_estimator(self, observations, actions, rewards):
        self.estimator.update(observations, actions, rewards)


if __name__ == "__main__":
    seed_everything(1)
    explore = PerArmExploration(num_tasks=2, num_cls=2, num_slates=2)
    print(explore.num_chosens)
    print(explore.num_data_sent)
    obs = np.array([
        [0, 0],
        [0, 1],
        #[1, 0],
        [1, 1],
        [1, 1],
    ])
    print(explore.get_action(obs, np.array([-10, 1, 10, 50])))

    estimator = EmpiricalEstimator(num_tasks=2, num_cls=2)
    print(estimator.Q)
    estimator.update(obs, actions=np.array([0, 3]), rewards=np.array([1, -2]))
    print(estimator.Q)
    print(estimator.get_Q(obs))
    # action = np.array([2, 0])
    # explore.update(obs, action)
    # print(explore.num_data_sent)
