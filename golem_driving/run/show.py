from typing import Union
import pyglet
from pyglet.window import key

import gym
from gym_duckietown.simulator import Simulator

from golem_driving.agents.agent import Agent
from golem_driving.config import ShowConfig


def show(env: Union[gym.Wrapper, Simulator], agent: Agent, config: ShowConfig) -> type(None):
    obs = env.reset()
    env.render()

    handler = key.KeyStateHandler()
    window = env.unwrapped.window
    window.push_handlers(handler)

    def update(dt):
        nonlocal obs
        a = agent.act(obs)
        obs, reward, done, _ = env.step(a)

        if done:
            obs = env.reset()

        env.render()

        if handler[key.ESCAPE]:
            env.close()
            pyglet.app.exit()

    # for manual agent:
    if hasattr(agent, 'set_key_handler'):
        agent.set_key_handler(handler)

    pyglet.clock.schedule_interval(update, interval=1.0 / env.unwrapped.frame_rate)

    pyglet.app.run()
