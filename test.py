import numpy as np


a = np.array(
    [[1738, 2137, 1955, 2073, 1925, 1282, 1027,  939,  820,  782, ],
     [1590, 1716, 1762, 1462,  988,  667,  667,  727,  661,  682, ]]
)
print(np.sum(a, axis=1))
