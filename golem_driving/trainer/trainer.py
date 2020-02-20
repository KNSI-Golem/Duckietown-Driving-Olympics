from gym_duckietown.simulator import Simulator

from golem_driving.agents.agent import Agent

from typing import Union

from numbers import Number

import numpy as np


class Trainer:
    def __init__(self, agent: Agent, env: Simulator):
        self.agent = agent
        self.env = env
    

    def register_obs(self, obs: np.ndarray, done: bool = True, reward: float = 0.0) -> None:
        """
        Learning
        """
        pass

    def model_step(self, obs: np.ndarray) -> Union[np.ndarray, Number]:
        """
        Generates new step
        """
        return self.agent.act(obs)
