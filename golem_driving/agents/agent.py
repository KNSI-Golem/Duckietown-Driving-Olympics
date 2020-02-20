from typing import Union
from numbers import Number
import numpy as np


class Agent:
    def act(self, obs: np.ndarray) -> Union[np.ndarray, Number]:
        action = np.array([0.44, 0.0])
        return action
