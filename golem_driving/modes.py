from typing import Callable

from golem_driving.run.show import show
from golem_driving.run.train import train
from golem_driving.run.test import test
import golem_driving.config as configs


class Mode(object):
    def __init__(self, run, config: Callable[[], configs.Config]):
        self.run = run
        self.config = config


modes = {
    'show': Mode(show, configs.ShowConfig),
    'train': Mode(train, configs.TrainConfig),
    'test': Mode(test, configs.TestConfig)
}
