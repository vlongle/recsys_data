import torch.nn as nn
import torch

lstm = nn.LSTM(5, 2)

x = torch.randn(1, 3, 5)
out, (h, c) = lstm(x)
print(out)
print(c)
hidden = (torch.ones(1, 3, 2), torch.ones(1, 3, 2))
out2, (h2, c2) = lstm(x, hidden)
print(out2)
print(c2)
