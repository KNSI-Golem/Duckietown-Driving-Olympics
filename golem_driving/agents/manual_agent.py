import numpy as np
from pyglet.window import key
from numbers import Number
from typing import Any, Union, Mapping

from golem_driving.agents.agent import Agent


class ManualAgent(Agent):
    def __init__(self):
        self.key_handler = None

    def act(self, obs: np.ndarray) -> Union[np.ndarray, Number]:
        action = np.array([0.0, 0.0])
        if self.key_handler[key.UP]:
            action = np.array([0.44, 0.0])
        if self.key_handler[key.DOWN]:
            action = np.array([-0.44, 0])
        if self.key_handler[key.LEFT]:
            action = np.array([0.35, +1])
        if self.key_handler[key.RIGHT]:
            action = np.array([0.35, -1])
        if self.key_handler[key.SPACE]:
            action = np.array([0, 0])

        return action

    def set_key_handler(self, key_handler: Mapping[int, Any]) -> type(None):
        self.key_handler = key_handler
