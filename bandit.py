import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning_lite.utilities.seed import seed_everything
import numpy as np
from utils import get_batch_tasks_cls
from abc import ABC, abstractmethod
from typing import Optional, Tuple, Dict, List


class Estimator:
    @abstractmethod
    def get_Q(self, observations: np.ndarray, eval=True):
        pass

    @abstractmethod
    def update(self, observations: np.ndarray, actions: np.ndarray, rewards: np.ndarray):
        pass


class ExplorationStrategy:
    @abstractmethod
    def get_action(self, observations: np.ndarray, estimator: Estimator):
        pass

    @abstractmethod
    def update(self, observations: np.ndarray, actions: np.ndarray):
        pass


class EmpiricalEstimator(Estimator):
    def __init__(self, num_tasks: int, num_cls: int):
        self.num_tasks = num_tasks
        self.num_cls = num_cls
        self.cum_rewards = np.zeros((num_tasks, num_cls))
        self.num_chosens = np.zeros((num_tasks, num_cls))

    @property
    def Q(self) -> np.ndarray:
        """
        Compute the Q values.
        Returns:
            Q: (num_tasks, num_cls) array of Q values.
        """
        return self.cum_rewards / np.maximum(self.num_chosens, 1)

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
                          np.maximum(self.num_chosens[tasks, cls], 1)) * self.epsilon
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


class NeuralEstimator(Estimator):
    def __init__(self, num_tasks: int, num_cls: int):
        self.num_tasks = num_tasks
        self.num_cls = num_cls
        # given a task and a class, predict the Q value
        self.model = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        print("Estimator num params: ", sum(p.numel()
              for p in self.model.parameters()))

    def get_Q(self, observations: np.ndarray, eval: Optional[bool] = True) -> np.ndarray:
        if eval:
            self.model.eval()
            with torch.no_grad():
                Q = self.model(torch.from_numpy(
                    observations).float()).squeeze().numpy()
        else:
            self.model.train()
            Q = self.model(torch.from_numpy(
                observations).float()).squeeze()
        return Q

    @property
    def Q(self, eval=True) -> np.ndarray:
        return self.get_Q(np.array([[task, c] for task in range(self.num_tasks) for c in range(self.num_cls)]), eval=True).reshape(self.num_tasks, self.num_cls)

    def update(self, observations: np.ndarray, actions: np.ndarray, rewards: np.ndarray):
        pred_Q = self.get_Q(observations, eval=False)[actions]
        loss = self.criterion(pred_Q, torch.from_numpy(rewards).float())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


class RecurrentNeuralEstimator(Estimator):
    """
    To estimate the rewards, we also take into account the previous data that we already sent to the user.
    TODO: Should consult recurrent DQN (https://mlpeschl.com/post/tiny_adrqn/)
    """

    def __init__(self, num_tasks: int, num_cls: int):
        self.num_tasks = num_tasks
        self.num_cls = num_cls
        # use a LSTM to predict the Q value
        self.internal_state_model = nn.LSTM(2, 32)
        self.model = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        self.hiddens = None
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(
            list(self.internal_state_model.parameters()) + list(self.model.parameters()), lr=0.001)

        print("Estimator Num params", sum(p.numel()
              for p in list(self.model.parameters()) + list(self.internal_state_model.parameters()) if p.requires_grad))

    def forward(self, x, batch_first=True):
        """
        Args:
            x: (batch_size, 2) array of (task, class) tuples where batch_first is True.
            x: (seq_len, 2) array of (task, class) tuples where batch_first is False. This
            is for updating the internal state.
        """
        # convert x to (seq_len=1, batch_size, 2)
        x
        if batch_first:
            x = x.unsqueeze(0)
        else:
            x = x.unsqueeze(1)
        hiddens = self.hiddens
        if batch_first and hiddens is not None:
            # replicate the hidden state for the new batch
            hiddens = (hiddens[0].repeat(1, x.shape[1], 1),
                       hiddens[1].repeat(1, x.shape[1], 1))
        out, hiddens = self.internal_state_model(x, hiddens)
        return self.model(out.squeeze()), hiddens

    def get_Q(self, observations: np.ndarray, eval: Optional[bool] = True) -> np.ndarray:
        if eval:
            self.model.eval()
            with torch.no_grad():
                Q, _ = self.forward(torch.from_numpy(
                    observations).float())
                Q = Q.squeeze().numpy()
        else:
            self.model.train()
            Q, _ = self.forward(torch.from_numpy(
                observations).float())
            Q = Q.squeeze()
        return Q

    @property
    def Q(self) -> np.ndarray:
        return self.get_Q(np.array([[task, c] for task in range(self.num_tasks) for c in range(self.num_cls)]), eval=True).reshape(self.num_tasks, self.num_cls)

    def update(self, observations: np.ndarray, actions: np.ndarray, rewards: np.ndarray):
        pred_Q = self.get_Q(observations, eval=False)[actions]
        loss = self.criterion(pred_Q, torch.from_numpy(rewards).float())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        # update hidden state by feeding observations[actions] to the LSTM
        with torch.no_grad():
            _, self.hiddens = self.forward(torch.from_numpy(
                observations[actions]).float(), batch_first=False)


class Algorithm:
    def __init__(self, estimator: Estimator, exploration_strategy: ExplorationStrategy):
        self.estimator = estimator
        self.exploration_strategy = exploration_strategy

    def predict(self, obs: np.ndarray):
        """
        obs: (batch_size, num_features) array of observations.
        """
        Q_values = self.estimator.get_Q(obs, eval=True)
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
