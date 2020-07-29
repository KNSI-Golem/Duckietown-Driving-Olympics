from typing import Union
from numbers import Number

import numpy as np

from golem_driving.utils.serializable import Serializable


class Agent(Serializable):
    def act(self, obs: np.ndarray) -> Union[np.ndarray, Number]:
        action = np.array([0.44, 0.0])
        return action
