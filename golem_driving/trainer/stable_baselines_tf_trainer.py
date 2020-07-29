import json
from numbers import Number
from typing import Any, Union
import zipfile

import numpy as np
import stable_baselines
from gym_duckietown.simulator import Simulator

from golem_driving.agents.agent import Agent
from golem_driving.trainer.trainer import Trainer


def make_policy(**kwargs):
    raise NotImplementedError


class StableBaselinesTFTrainer(Trainer):
    """
    Note: some algorithms may require mpi dll, check stable-baselines installation guide
    """
    def __init__(self, env: Simulator, algo: str, policy: Union[str, dict], iter_steps: int=100000,
                 model: stable_baselines.common.BaseRLModel=None, **kwargs):
        super(StableBaselinesTFTrainer, self).__init__(env)

        self.iter_steps = iter_steps
        self.algo = algo

        if model is not None:
            self.model = model
        else:
            policy = policy if isinstance(policy, str) else make_policy(**policy)
            self.model = getattr(stable_baselines, algo)(policy, env, **kwargs)

    def train_iter(self) -> None:
        """
        Learning
        """
        self.model.learn(self.iter_steps)

    def act(self, obs: np.ndarray) -> Union[np.ndarray, Number]:
        return self.model.predict(obs, deterministic=True)[0]

    def set_env(self, env: Simulator) -> None:
        super(StableBaselinesTFTrainer, self).set_env(env)
        self.model.set_env(env)
    
    @staticmethod
    def load(filename: str) -> Any:
        with zipfile.ZipFile(filename, 'r') as f:
             data = json.loads(f.read('golemdata.json').decode('utf-8'))
        iter_steps = data['iter_steps']
        algo = data['algo']
        model = getattr(stable_baselines, algo).load(filename)

        return StableBaselinesTFTrainer(env=None, algo=algo, policy=None, iter_steps=iter_steps, model=model)

    def save(self, filename: str) -> None:
        self.model.save(filename)
        
        with zipfile.ZipFile(filename, 'a') as f:
            data = json.dumps({
                'iter_steps': self.iter_steps,
                'algo': self.algo})
            f.writestr('golemdata.json', data)
