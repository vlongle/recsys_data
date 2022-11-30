from preference_estimator import (
    Estimator,
    EmpiricalEstimator,
)
from exploration_strategy import (
    ExplorationStrategy,
    PerArmExploration,
)
from lightning_lite.utilities.seed import seed_everything
import numpy as np
from typing import (
    List,
)


class BanditAlgorithm:
    def __init__(self, estimator: Estimator, exploration_strategy: ExplorationStrategy):
        self.estimator = estimator
        self.exploration_strategy = exploration_strategy

    def predict(self, obs: np.ndarray):
        """
        obs: (batch_size, num_features) array of observations.
        """
        # print("num_chosens:")
        # print(self.estimator.num_chosens)
        # print("Q:")
        # print(self.estimator.Q)
        Q_values = self.estimator.get_Q(obs, eval=True)
        action = self.exploration_strategy.get_action(obs, Q_values)
        self.exploration_strategy.update(obs, action)
        return action

    def update_estimator(self, observations, actions, rewards, update_batch_size=None):
        if update_batch_size is not None:
            observations, actions, rewards = self.preprocess(observations,
                                                             actions,
                                                             rewards,
                                                             update_batch_size)

        self.estimator.update(observations, actions, rewards)
        return

    def preprocess(self, observations: List[np.ndarray], actions, rewards, update_batch_size):
        assert len(observations) == len(actions) == len(
            rewards) == update_batch_size, "update_batch_size must be equal to the number of observations, actions, and rewards"

        batch_sizes = [obs.shape[0] for obs in observations]
        # concat all observations, actions, rewards
        observations = np.concatenate(observations, axis=0)
        cum_batch_sizes = np.cumsum(batch_sizes)
        # process actions to increment to the correct index
        for batch_idx in range(1, len(cum_batch_sizes)):
            actions[batch_idx] += cum_batch_sizes[batch_idx - 1]

        # BUG: rewards concat like this is WRONG!!!!!
        actions = np.concatenate(actions, axis=0)
        rewards = np.concatenate(rewards, axis=0)

        return observations, actions, rewards


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
