import functools
import importlib
import os
import pickle
from typing import Any, Callable, Mapping, Tuple, Optional, IO
import yaml

import gym

from golem_driving.agents.agent import Agent


class Config(object):
    def __init__(self):
        self.agent = None
        self.agent_file = None
        self.env_wrappers = None

    def _load_object(self, obj_config: Mapping[str, Any]) -> Tuple[Callable, Optional[str]]:
        builder = importlib.import_module(obj_config['module']).__dict__[obj_config['builder']]
        args = obj_config.get('args', [])
        kwargs = obj_config.get('kwargs', {})
        file = obj_config.get('file', None)

        return functools.partial(builder, *args, **kwargs), file

    def build_agent(self) -> Agent:
        if self.agent_file and os.path.isfile(self.agent_file):
            with open(self.agent_file, 'rb') as stored_agent_file:
                return pickle.load(stored_agent_file)
        return self.agent()

    def _load_base(self, file: Mapping[str, Any]) -> type(None):
        self.agent, self.agent_file = self._load_object(file['agent'])

        self.env_wrappers =\
            [self._load_object(wrapper)[0] for wrapper in file['env_wrappers']]\
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
        self.trainer = None
        self.steps = None
        self.save_agent = None

        super(TrainConfig, self).__init__()

    def build_trainer(self, agent: Agent, env: gym.Env) -> Any:
        return self.trainer(agent, env)

    def _load(self, file: Mapping[str, Any]) -> type(None):
        self.trainer = self._load_object(file['trainer'])
        self.steps = file.get('steps', 1e6)
        self.save_agent = file.get('save_agent', True)
