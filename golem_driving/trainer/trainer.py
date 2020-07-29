import numpy as np
from gym_duckietown.simulator import Simulator

from golem_driving.agents.agent import Agent


class Trainer(Agent):
    def __init__(self, env: Simulator):
        self.env = env
    

    def train_iter(self) -> None:
        """
        Learning
        """
        pass

    def set_env(self, env: Simulator) -> None:
        self.env = env
