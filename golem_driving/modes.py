from golem_driving.run.show import show
from golem_driving.run.train import train
import golem_driving.config as configs


class Mode(object):
    def __init__(self, run, config):
        self.run = run
        self.config = config


modes = {
    'show': Mode(show, configs.ShowConfig)
    'train': Mode(train, configs.TrainConfig)
}
