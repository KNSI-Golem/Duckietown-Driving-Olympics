from typing import Callable, Union

from gym_duckietown.simulator import Simulator

import golem_driving.config as configs
from golem_driving.agents.agent import Agent
from golem_driving.run.show import show
from golem_driving.run.train import train
from golem_driving.run.test import test
from golem_driving.trainer.trainer import Trainer

class Mode(object):
    def __init__(self, run, config: Callable[[], configs.Config], use_trainer: bool):
        self._run = run
        self.config = config
        self.use_trainer = use_trainer
    
    def run(self, env: Simulator, agent: Union[Agent, Trainer], config: configs.Config):
        agent_type = Trainer if self.use_trainer else Agent
        assert isinstance(agent, agent_type)

        self._run(env, agent, config)


modes = {
    'show': Mode(show, configs.ShowConfig, False),
    'train': Mode(train, configs.TrainConfig, True),
    'test': Mode(test, configs.TestConfig, False)
}
