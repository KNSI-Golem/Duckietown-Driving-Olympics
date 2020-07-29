import pickle
from typing import Union

import gym
from gym_duckietown.simulator import Simulator

from golem_driving.trainer.trainer import Trainer
from golem_driving.config import TrainConfig


def train(env: Union[gym.Wrapper, Simulator], trainer: Trainer, config: TrainConfig) -> type(None):
    for i in range(config.iters):
        print(f'Iteration {i + 1}\tof {config.iters}')
        trainer.train_iter()
    print('Done')

    if config.agent_file:
        print('Saving agent...')
        trainer.save(config.agent_file)
