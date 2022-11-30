import torch.nn as nn
import torch
from copy import deepcopy
import numpy as np


class MNISTNet(nn.Module):
    def __init__(self, X_test, y_test) -> None:
        super().__init__()
        self.num_classes_per_task = 10
        self.num_tasks = 3
        self.num_classes = self.num_classes_per_task * self.num_tasks
        self.embedding = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 4, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.model = nn.Sequential(
            nn.Linear(4 * 7 * 7, 64),
            nn.ReLU(),
            nn.Linear(64, self.num_classes),
        )

        self.criteria = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        print("MNIST num parameters:", sum(p.numel()
              for p in self.parameters()))
        self.X_test = X_test.to('cuda')
        self.y_test = y_test.to('cuda')
        self.to('cuda')

    def forward(self, x):
        x = x.to('cuda')
        x = self.embedding(x)
        x = x.view(-1, 4 * 7 * 7)
        x = self.model(x)
        return x

    def train_step(self, x, y):
        y = y.to('cuda')
        self.train()
        self.optimizer.zero_grad()
        y_hat = self(x)
        loss = self.criteria(y_hat, y)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def test_step(self):
        self.eval()
        with torch.no_grad():
            y_hat = self(self.X_test)
            loss = self.criteria(y_hat, self.y_test)
            return loss.item()

    def test_acc(self):
        self.eval()
        with torch.no_grad():
            y_hat = self(self.X_test)
            acc = (y_hat.argmax(dim=1) == self.y_test).float().mean().item()
            return acc

    def evaluate_usefulness_each_one(self, X, ys):
        # for each (x, y) in X, ys, evaluate the usefulness of the point
        # slow and NOISY
        batch_size = X.shape[0]
        before_loss = self.test_step()
        losses = []
        # TODO: parallelize this
        for i in range(batch_size):
            x = X[i]
            y = ys[i]
            net_copy = deepcopy(self)
            x = x.unsqueeze(0)
            y = y.unsqueeze(0)
            net_copy.train_step(x, y)
            losses.append(net_copy.test_step())
        return before_loss - np.array(losses)
        # return (before_loss - np.array(losses)) / before_loss

    def evaluate_usefulness_leave_one_out(self, X, ys):
        # leave one out: less noisy (operable), still slow.
        batch_size = X.shape[0]
        net_copy = deepcopy(self)
        net_copy.train_step(X, ys)
        fitted_all = net_copy.test_step()
        improvs = []
        # TODO: parallelize this
        for i in range(batch_size):
            net_copy_leave_one = deepcopy(self)
            # leave data point i out and fit the model
            # on the rest of the data
            X_leave_one = torch.cat([X[:i], X[i+1:]])
            ys_leave_one = torch.cat([ys[:i], ys[i+1:]])
            net_copy_leave_one.train_step(X_leave_one, ys_leave_one)
            fitted_leave_one = net_copy_leave_one.test_step()
            improvs.append(fitted_leave_one - fitted_all)
        # return np.array(improvs)
        return np.array(improvs) / max(improvs)

    def evaluate_usefulness_uniform(self, batch_x, batch_y):
        batch_z = batch_y.clone()
        batch_z.apply_(lambda x: x // 10)
        before_loss = self.test_step()

        rewards = np.zeros(batch_x.shape[0])
        test_loss_task_list = {}  # for DEBUG
        for task in range(self.num_tasks):
            net_copy_task = deepcopy(self)
            batch_x_task = batch_x[batch_z == task]
            batch_y_task = batch_y[batch_z == task]
            net_copy_task.train_step(batch_x_task, batch_y_task)
            test_loss_task = net_copy_task.test_step()
            rewards[batch_z == task] = before_loss - test_loss_task
            test_loss_task_list[task] = test_loss_task

        return rewards
