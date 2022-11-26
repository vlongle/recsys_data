import numpy as np
from typing import Tuple


def get_batch_tasks_cls(observations: np.ndarray, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    routed_batch = observations[actions]
    batch_tasks = routed_batch[:, 0]
    batch_cls = routed_batch[:, 1]
    return batch_tasks, batch_cls
