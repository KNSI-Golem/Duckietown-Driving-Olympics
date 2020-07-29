import functools
import importlib
import os
import pickle
from typing import Any, Callable, Mapping, Tuple, Optional, IO, Union
import yaml

import gym
from gym_duckietown.simulator import Simulator

from golem_driving.agents.agent import Agent
from golem_driving.trainer.trainer import Trainer


class Config(object):
    def __init__(self):
        self.agent = None
        self.agent_loader = None
        self.agent_file = None
        self.env_wrappers = None

    def _load_object(self, obj_config: Mapping[str, Any]) -> Tuple[Callable, Callable, Optional[str]]:
        builder = importlib.import_module(obj_config['module']).__dict__[obj_config['builder']]
        args = obj_config.get('args', [])
        kwargs = obj_config.get('kwargs', {})

        return functools.partial(builder, *args, **kwargs)

    def _load_serializable_object(self, obj_config: Mapping[str, Any]) -> Tuple[Callable, Callable, Optional[str]]:
        if 'module' in obj_config and 'builder' in obj_config:
            builder = importlib.import_module(obj_config['module']).__dict__[obj_config['builder']]
            args = obj_config.get('args', [])
            kwargs = obj_config.get('kwargs', {})

            builder_fn = functools.partial(builder, *args, **kwargs)
            loader = builder.load
        else:
            builder_fn = None
            loader = None

        file = obj_config.get('file', None)

        return builder_fn, loader, file

    def build_agent(self, *args, **kwargs) -> Agent:
        if self.agent_file and os.path.isfile(self.agent_file):
            print('Loading agent...')
            return self.agent_loader(self.agent_file)
        print('Creating agent...')
        return self.agent(*args, **kwargs)

    def _load_base(self, file: Mapping[str, Any]) -> type(None):
        self.agent, self.agent_loader, self.agent_file =\
            self._load_serializable_object(file['agent']) if 'agent' in file else (None, None, None)

        self.env_wrappers =\
            [self._load_object(wrapper) for wrapper in file['env_wrappers']]\
            if 'env_wrappers' in file else []

    def _load(self, file: Mapping[str, Any]) -> type(None):
        raise NotImplementedError

    def load(self, file: IO) -> type(None):
        file = yaml.full_load(file)
        self._load_base(file)
        self._load(file)


class ShowConfig(Config):
    def __init__(self):
        super(ShowConfig, self).__init__()

    def _load(self, file: Mapping[str, Any]) -> type(None):
        pass


class TrainConfig(Config):
    def __init__(self):
        self.iters = None
        super(TrainConfig, self).__init__()

    def build_trainer(self, env: Union[Simulator, gym.Wrapper]) -> Trainer:
        trainer = self.build_agent(env=env)

        if trainer.env is None:
            trainer.set_env(env)

        return trainer
        
    def _load(self, file: Mapping[str, Any]) -> type(None):
        self.iters = file.get('iters', 1)


class TestConfig(Config):
    def __init__(self):
        self.episodes = None

        super(TestConfig, self).__init__()

    def _load(self, file: Mapping[str, Any]) -> type(None):
        self.episodes = file.get('episodes', 10)
