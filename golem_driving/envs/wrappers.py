from typing import Any, Union, Iterable, Tuple, Sequence
from numbers import Number

import gym
from gym.spaces import Box
import numpy as np
from skimage.transform import resize

from gym_duckietown.simulator import Simulator


class ScalingWrapper(gym.Wrapper):
    def __init__(self, env: Union[gym.Wrapper, Simulator], target_size: Sequence[int]):
        super(ScalingWrapper, self).__init__(env)
        self.target_shape = (*target_size, *self.observation_space.shape[len(target_size):])

        self.observation_space.shape = Box(
            np.min(self.observation_space.low),
            np.max(self.observation_space.high),
            self.target_shape,
            self.observation_space.dtype
        )

    def scale(self, obs: np.array) -> np.ndarray:
        return resize(obs, self.target_shape, mode='edge')

    def reset(self) -> np.ndarray:
        obs = self.env.reset()
        return self.scale(obs)

    def step(self, action: Union[Number, Iterable[Number]]) -> Tuple[np.ndarray, Number, bool, Any]:
        obs, reward, done, misc = self.env.step(action)
        return self.scale(obs), reward, done, misc
