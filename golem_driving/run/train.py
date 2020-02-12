import pickle
from typing import Union

import gym
from gym_duckietown.simulator import Simulator

from golem_driving.agents.agent import Agent
from golem_driving.config import TrainConfig


def train(env: Union[gym.Wrapper, Simulator], agent: Agent, config: TrainConfig) -> type(None):
    obs = env.reset()
    trainer = config.build_trainer(agent, env)
    trainer.register_obs(obs)

    for _ in range(config.steps):
        act = trainer.model_step(obs)
        obs, reward, done, info = env.step(act)

        trainer.register_obs(obs, done, reward)

        if done:
            obs = env.reset()
            trainer.register_obs(obs)

    if config.save_agent and config.agent_file:
        with open(config.agent_file, 'wb') as agent_file:
            pickle.dump(agent, agent_file)
