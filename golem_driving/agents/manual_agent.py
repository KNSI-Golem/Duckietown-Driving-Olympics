import numpy as np
from pyglet.window import key


class ManualAgent(object):
    def __init__(self):
        self.key_handler = None

    def act(self, obs):
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

    def set_key_handler(self, key_handler):
        self.key_handler = key_handler
