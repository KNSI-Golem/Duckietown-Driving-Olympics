from typing import Union
from numbers import Number
import numpy as np


class Agent:
    def act(self, obs: np.ndarray) -> Union[np.ndarray, Number]:
        raise NotImplementedError
