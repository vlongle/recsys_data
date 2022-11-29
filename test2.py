import numpy as np
batch_sizes = [4, 2, 6]
actions = [np.array([0, 2]), np.array([1, 0]), np.array([5, 3])]
cum_batch_sizes = np.cumsum(batch_sizes)

for batch_idx in range(1, len(cum_batch_sizes)):
    actions[batch_idx] += cum_batch_sizes[batch_idx-1]

print(actions)
