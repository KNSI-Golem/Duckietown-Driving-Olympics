import functools
import importlib
import yaml


class Config(object):
    def __init__(self):
        self.agent = None

    def _load_object(self, obj_config):
        builder = importlib.import_module(obj_config['module']).__dict__[obj_config['builder']]
        args = obj_config.get('args', [])
        kwargs = obj_config.get('kwargs', {})

        return functools.partial(builder, *args, **kwargs)

    def build_agent(self):
        return self.agent()

    def _load_base(self, file):
        self.agent = self._load_object(file['agent'])

    def _load(self, file):
        raise NotImplementedError

    def load(self, file):
        file = yaml.full_load(file)
        self._load_base(file)
        self._load(file)


class ShowConfig(Config):
    def __init__(self):
        super(ShowConfig, self).__init__()

    def _load(self, file):
        pass


class TrainConfig(Config):
    def __init__(self):
        self.trainer = None
        self.steps = None

        super(TrainConfig, self).__init__()

    def build_trainer(self, agent, env):
        return self.trainer(agent, env)

    def _load(self, file):
        self.trainer = self._load_object(file['trainer'])
        self.steps = file.get('steps', 1e6)
