from abc import abstractmethod
import numpy as np
from utils import get_batch_tasks_cls
import torch.nn as nn
from typing import Optional
import torch

from rich.console import Console


class Estimator:
    def __init__(self, num_tasks: int, num_cls: int, use_img=False, cfg={}):
        pass

    @abstractmethod
    def get_Q(self, observations: np.ndarray, eval=True):
        pass

    @abstractmethod
    def update(self, observations: np.ndarray, actions: np.ndarray, rewards: np.ndarray):
        pass


class DummyEstimator(Estimator):
    def get_Q(self, observations: np.ndarray, eval=True):
        return np.zeros(observations.shape[0])

    def update(self, observations: np.ndarray, actions: np.ndarray, rewards: np.ndarray):
        pass


class EmpiricalEstimator(Estimator):
    def __init__(self, num_tasks: int, num_cls: int, use_img=False, cfg={}):
        assert use_img == False, "EmpiricalEstimator does not support image."
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
        delta = np.zeros((self.num_tasks, self.num_cls))
        batch_cls = batch_cls % self.num_cls
        for task in range(self.num_tasks):
            for c in range(self.num_cls):
                delta[task, c] = np.sum(
                    ((batch_tasks == task) & (batch_cls == c)) * rewards)
                self.num_chosens[task, c] += np.sum(
                    (batch_tasks == task) & (batch_cls == c))

        print("delta:")
        print(delta)
        self.cum_rewards += delta

    def get_Q(self, observations: np.ndarray, eval=True):
        """
        Args:
            observations: (batch_size, 2) array of (task, class) tuples.
        """
        batch_tasks, batch_cls = observations[:, 0], observations[:, 1]
        # map batch_cls to lambda x: x % self.num_cls
        batch_cls = batch_cls % self.num_cls
        return self.Q[batch_tasks, batch_cls]


class NeuralEstimator(Estimator):
    def __init__(self, num_tasks: int, num_cls: int, use_img: Optional[bool] = False, cfg={}):
        self.num_tasks = num_tasks
        self.num_cls = num_cls
        self.use_img = use_img

        if self.use_img:
            self.encoder = nn.Sequential(
                nn.Conv2d(1, 16, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
                nn.Conv2d(16, 4, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
            )  # 28x28 -> 4x7x7
            self.extra_encoder = nn.Linear(4 * 7 * 7, 32)
        else:
            self.encoder = nn.Embedding(num_cls * num_tasks, 32,
                                        max_norm=2,
                                        norm_type=2)
            # self.encoder = nn.Sequential(
            #     nn.Linear(2, 32),
            #     nn.ReLU(),
            # )
        # given a task and a class, predict the Q value
        self.model = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )
        self.criterion = nn.MSELoss()
        params = list(self.encoder.parameters()) + \
            list(self.model.parameters())
        if self.use_img:
            params += list(self.extra_encoder.parameters())

        self.optimizer = torch.optim.Adam(params, lr=0.001)
        print("Neural Estimator num params: ", sum(p.numel()
              for p in params))

    def forward(self, x):
        if not self.use_img:
            # (batch_size, 2) -> (batch_size,) array of task * num_cls + class
            x = x[:, 1].long()
        x = self.encoder(x)
        if not self.use_img:
            x = nn.ReLU()(x)
        if self.use_img:
            x = x.view(-1, 4 * 7 * 7)
            x = self.extra_encoder(x)
        return self.model(x)

    def get_Q(self, observations: np.ndarray, eval: Optional[bool] = True) -> np.ndarray:
        if eval:
            self.model.eval()
            with torch.no_grad():
                Q = self.forward(torch.from_numpy(
                    observations).float()).squeeze().numpy()
        else:
            self.model.train()
            Q = self.forward(torch.from_numpy(
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
    Use the internal state concatenated with the observations to predict the Q value.
    """

    def __init__(self, num_tasks: int, num_cls: int, use_img=False, cfg={}):
        self.num_tasks = num_tasks
        self.num_cls = num_cls
        self.use_img = use_img
        # use a LSTM to predict the Q value
        self.state_embedder = nn.Linear(2, 32)
        self.internal_state_model = nn.LSTM(2, 32)
        self.model = nn.Sequential(
            nn.Linear(32 + 32, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        self.hiddens = None
        self.last_user_state = torch.zeros(32)
        self.criterion = nn.MSELoss()

        params = list(self.internal_state_model.parameters()) + \
            list(self.model.parameters()) + \
            list(self.state_embedder.parameters())
        print("Recurrent Estimator (concat) Num params", sum(p.numel()
              for p in params if p.requires_grad))
        self.optimizer = torch.optim.Adam(
            params, lr=0.001)

    def forward(self, x):
        """
        Args:
            x: (batch_size, 2) array of (task, class) tuples where batch_first is True.
        """
        # replicate self.user_state to (batch_size, 32)
        user_state = self.last_user_state.repeat(x.shape[0], 1)
        # concatenate user_state and x
        x = self.state_embedder(x)
        x = torch.cat([user_state, x], dim=1)
        return self.model(x)

    def get_Q(self, observations: np.ndarray, eval: Optional[bool] = True) -> np.ndarray:
        if eval:
            self.model.eval()
            with torch.no_grad():
                Q = self.forward(torch.from_numpy(
                    observations).float()).squeeze().numpy()
        else:
            self.model.train()
            Q = self.forward(torch.from_numpy(
                observations).float()).squeeze()
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
        # update self.last_user_state by feeding observations[actions] to the LSTM
        user_states, hiddens = self.internal_state_model(
            torch.from_numpy(observations[actions]).float(), self.hiddens)
        self.hiddens = (hiddens[0].detach(), hiddens[1].detach())
        self.last_user_state = user_states[-1]


class RecurrentNeuralEstimatorV0(Estimator):
    """
    To estimate the rewards, we also take into account the previous data that we already sent to the user.
    Feed all the observations into LSTM to predict the Q value.
    """

    def __init__(self, num_tasks: int, num_cls: int, use_img: Optional[bool] = False, cfg={}):
        self.num_tasks = num_tasks
        self.num_cls = num_cls
        self.use_img = use_img
        if self.use_img:
            self.encoder = nn.Sequential(
                nn.Conv2d(1, 16, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
                nn.Conv2d(16, 4, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
            )  # 28x28 -> 4x7x7
            self.internal_state_model = nn.LSTM(4 * 7 * 7, 64)
        else:
            # self.encoder = nn.Linear(2, 32)
            self.encoder = nn.Embedding(num_cls * num_tasks, 32,
                                        max_norm=2,
                                        norm_type=2)
            # use a LSTM to predict the Q value
            self.internal_state_model = nn.LSTM(32, 64)
        self.model = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        self.hiddens = None
        self.criterion = nn.MSELoss()
        params = list(self.model.parameters()) + \
            list(self.internal_state_model.parameters()) + \
            list(self.encoder.parameters())
        self.optimizer = torch.optim.Adam(params, lr=0.001)

        self.step = 0
        self.reset_period = cfg.get("reset_period", 100000)

        self.console = Console()
        print("Recurrent Estimator (one-feed) Num params", sum(p.numel()
              for p in params if p.requires_grad))

    def forward(self, x, batch_first=True):
        """
        Args:
            x: (batch_size, 2) array of (task, class) tuples where batch_first is True.
            x: (seq_len, 2) array of (task, class) tuples where batch_first is False. This
            is for updating the internal state.
        """
        # convert x to (seq_len=1, batch_size, 2)
        if not self.use_img:
            # (batch_size, 2) -> (batch_size,) array of task * num_cls + class
            x = x[:, 1].long()
        x = self.encoder(x)
        if self.use_img:
            x = x.view(-1, 4 * 7 * 7)
        if batch_first:
            x = x.unsqueeze(0)
        else:
            x = x.unsqueeze(1)
        hiddens = self.hiddens
        # print("forward hidden", hiddens)
        if batch_first and hiddens is not None:
            # replicate the hidden state for the new batch
            hiddens = (hiddens[0].repeat(1, x.shape[1], 1),
                       hiddens[1].repeat(1, x.shape[1], 1))
        out, hiddens = self.internal_state_model(x, hiddens)
        # print("hiddens:", hiddens[0])
        # out2, hiddens2 = self.internal_state_model(x)
        # print("hiddens2:", hiddens2[0])
        # print("out:", out)
        # print("out2:", out2)
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

    @ property
    def Q(self) -> np.ndarray:
        return self.get_Q(np.array([[task, c] for task in range(self.num_tasks) for c in range(self.num_cls)]), eval=True).reshape(self.num_tasks, self.num_cls)

    def update(self, observations: np.ndarray, actions: np.ndarray, rewards: np.ndarray):
        if self.step % self.reset_period == 0:
            self.console.print(">> RESET", style="red")
            self.hiddens = None
        pred_Q = self.get_Q(observations, eval=False)[actions]
        loss = self.criterion(pred_Q, torch.from_numpy(rewards).float())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # update hidden state by feeding observations[actions] to the LSTM
        with torch.no_grad():
            _, self.hiddens = self.forward(torch.from_numpy(
                observations[actions]).float(), batch_first=False)

        self.step += 1
