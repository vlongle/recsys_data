import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt


mdataset = torchvision.datasets.MNIST(
    root='./data', train=True, download=True, transform=torchvision.transforms.ToTensor())
fdataset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=torchvision.transforms.ToTensor(),
                                             target_transform=lambda x: x + 10)

idx = torch.randperm(10)
fdataset[[0, 1, 2]]
