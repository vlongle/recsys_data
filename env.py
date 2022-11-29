
import torch
import torchvision
from lightning_lite.utilities.seed import seed_everything
import gym
from typing import (
    Optional,
    Tuple,
    Dict,
    List,
)

import numpy as np
from utils import get_batch_tasks_cls
from model import MNISTNet


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


class ImgRouterEnv(RouterEnv):
    def __init__(self, existing_samples: np.ndarray, target_samples: np.ndarray, cfg: Optional[Dict] = {}) -> None:
        self.mdataset = torchvision.datasets.MNIST(
            root='./data', train=True, download=True, transform=torchvision.transforms.ToTensor())
        self.fdataset = torchvision.datasets.FashionMNIST(
            root='./data', train=True, download=True, transform=torchvision.transforms.ToTensor(),
            target_transform=lambda x: x + 10)
        self.dataset = torch.utils.data.ConcatDataset(
            [self.mdataset, self.fdataset])

        self.exist_samples = existing_samples
        self.target_samples = target_samples
        self.num_candidates = cfg.get("num_candidates", 10)
        self.num_slates = cfg.get("num_slates", 2)
        self.num_tasks = 2
        self.num_classes = 10
        self.num_samples = len(self.dataset)
        self.max_env_steps = self.num_samples // self.num_candidates
        self.use_image = False

    def _get_full_obs(self) -> List[Tuple[torch.Tensor, int]]:
        """
        Get the current batch of data.
        Returns:
            obs: (num_candidates, 2) array of (task, class) tuples.
        """
        idx = self.dataset_stream_idx[self.env_step * self.num_candidates:
                                      (self.env_step + 1) * self.num_candidates]
        return [self.dataset[i] for i in idx]

    def _get_obs(self) -> np.ndarray:
        """
        Get the current batch of data.
        Returns:
            obs: (num_candidates, 2) array of (task, class) tuples.
        """
        full_obs = self._get_full_obs()
        if self.use_image:
            return self._get_image_obs(full_obs)
        else:
            return self._get_tc_obs(full_obs)

    def _get_image_obs(self, full_obs: List[Tuple[torch.Tensor, int]]) -> np.ndarray:
        """
        Get the current batch of data.
        Returns:
            obs: (num_candidates, 1, 28, 28) array of images where 1 is the channel dimension.
        """
        return torch.stack([img for img, _ in full_obs], dim=0).numpy()

    def _get_tc_obs(self, full_obs: List[Tuple[torch.Tensor, int]]) -> np.ndarray:
        obs = [(tc // 10, tc % 10) for _, tc in full_obs]
        # turns obs into a numpy array
        return np.array(obs)

    def reset(self):
        # self.data_stream_idx is a permutation of the indices of the dataset
        self.dataset_stream_idx = torch.randperm(len(self.dataset))
        self.env_step = 0
        self.current_samples = self.exist_samples.copy()
        return self._get_obs()


class ImgRouterEvalEnv(ImgRouterEnv):
    def __init__(self, cfg):
        mdataset_train = torchvision.datasets.MNIST(
            root='./data', train=True, download=True, transform=torchvision.transforms.ToTensor())
        fdataset_train = torchvision.datasets.FashionMNIST(
            root='./data', train=True, download=True, transform=torchvision.transforms.ToTensor(),
            target_transform=lambda x: x + 10)

        kdataset_train = torchvision.datasets.KMNIST(
            root='./data', train=True, download=True, transform=torchvision.transforms.ToTensor(),
            target_transform=lambda x: x + 20)

        self.dataset_train = torch.utils.data.ConcatDataset(
            [mdataset_train, fdataset_train, kdataset_train])

        mdataset_test = torchvision.datasets.MNIST(
            root='./data', train=False, download=True, transform=torchvision.transforms.ToTensor())

        self.X_test = torch.stack([x for x, _ in mdataset_test])
        self.y_test = torch.tensor([y for _, y in mdataset_test])
        self.batch_size = cfg.get("num_candidates", 32)
        self.num_slates = cfg.get("num_slates", 2)
        self.num_tasks = 3
        self.num_classes = 10
        self.max_steps = cfg.get("max_steps", np.inf)
        self.max_steps = min(self.max_steps, len(
            self.dataset_train) // self.batch_size)

        self.use_img = cfg.get("use_img", False)

        self.evaluate_strategy = cfg.get("evaluate_strategy", "uniform")

    def _get_full_obs(self) -> Tuple:
        batch_idx = self.dataset_stream_idx[self.env_step *
                                            self.batch_size:(self.env_step + 1) * self.batch_size]
        batch = [self.dataset_train[i] for i in batch_idx]
        batch_x = torch.stack([x for x, _ in batch])
        batch_y = torch.tensor([y for _, y in batch])
        batch_z = batch_y.clone()
        batch_z.apply_(lambda x: x // 10)
        # return batch_x.numpy(), batch_z.numpy(), batch_y.numpy()
        return batch_x, batch_z, batch_y

    def _get_obs(self) -> np.ndarray:
        """
        Returns:
            (batch_size, 2) array of (task, class) tuples.
        """
        batch_x, batch_z, batch_y = self._get_full_obs()
        batch_y.apply_(lambda x: x % 10)
        if self.use_img:
            return batch_x.numpy()
        else:
            return np.stack([batch_z.numpy(), batch_y.numpy()], axis=1)

    def reset(self):
        self.model = MNISTNet(self.X_test, self.y_test)
        if self.evaluate_strategy == "uniform":
            self.evaluate_fn = self.model.evaluate_usefulness_uniform
        elif self.evaluate_strategy == "leave_one_out":
            self.evaluate_fn = self.model.evaluate_usefulness_leave_one_out
        elif self.evaluate_strategy == "each_one":
            self.evaluate_fn = self.model.evaluate_usefulness_each_one
        else:
            raise ValueError("Invalid evaluate strategy")

        self.dataset_stream_idx = torch.randperm(len(self.dataset_train))
        self.env_step = 0
        self.current_samples = np.zeros((self.num_tasks, self.num_classes))
        self.routed_samples_per_time = np.zeros(
            (self.max_steps, self.num_tasks, self.num_classes))
        return self._get_obs()

    def step(self, action: np.ndarray) -> Tuple:
        batch_x, batch_z, batch_y = self._get_full_obs()
        routed_batch_x, routed_batch_y, routed_batch_z = batch_x[
            action], batch_y[action], batch_z[action]

        print("routed_batch_z", torch.unique(
            routed_batch_z, return_counts=True, sorted=True))
        rewards = self.evaluate_fn(
            routed_batch_x, routed_batch_y)

        # update states
        routed_batch_y.apply_(lambda x: x % 10)
        for task in range(self.num_tasks):
            for cls in range(self.num_classes):
                self.current_samples[task,
                                     cls] += torch.sum((routed_batch_z == task) & (routed_batch_y == cls)).item()
                self.routed_samples_per_time[self.env_step, task, cls] = torch.sum(
                    (routed_batch_z == task) & (routed_batch_y == cls)).item()

        self.env_step += 1
        self.model.train_step(routed_batch_x, routed_batch_y)
        done = self.env_step >= self.max_steps
        next_obs = np.array([]) if done else self._get_obs()

        return next_obs, np.sum(rewards), done, {"rewards": rewards}


if __name__ == "__main__":
    seed_everything(1)
    # existing_samples = np.array([[0, 0],
    #                             [10, 10]])
    # target_samples = np.array([[10, 10],
    #                            [10, 10]])

    # env = RouterEnv(existing_samples, target_samples,
    #                 cfg={"num_candidates": 5})
    # env.reset()
    # obs = env._get_obs()
    # print(obs)
    # env.step(np.array([0, 3]))

    num_samples = 60000 * 2

    reduce_fator = 32
    num_candidates = 64
    num_slates = num_candidates // reduce_fator

    num_tasks = 2
    num_cls = 10
    num_data_sent = num_samples // reduce_fator
    print("num_data_sent", num_data_sent)
    target_samples = np.ones((num_tasks, num_cls)) * (num_data_sent // num_cls)
    existing_samples = np.zeros((num_tasks, num_cls))
    existing_samples[1, :] = target_samples[1, :]
    print(target_samples)
    print(existing_samples)

    env = ImgRouterEnv(existing_samples, target_samples,
                       cfg={
                           "num_candidates": num_candidates,
                           "num_slates": num_slates,
                       })
    obs = env.reset()
    # batch_tasks, batch_cls = get_batch_tasks_cls(
    #     obs, np.arange(num_candidates))
    # print(np.unique(batch_tasks, return_counts=True))
    # print(np.unique(batch_cls, return_counts=True))
