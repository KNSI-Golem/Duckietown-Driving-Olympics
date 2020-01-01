import argparse
import os
import sys

# Add gym_duckietown to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'gym-duckietown'))

from golem_driving.modes import modes
from golem_driving.envs import add_env_args, get_env_from_args


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', required=True, type=str, choices=list(modes.keys()))
    parser.add_argument('--config', required=True, type=str)
    add_env_args(parser)
    args = parser.parse_args()

    mode = modes.get(args.mode)

    with open(args.config, 'r') as config_file:
        config = mode.config()
        config.load(config_file)

    with get_env_from_args(args, discrete=False) as env:
        agent = config.build_agent()

        mode.run(env=env, agent=agent, config=config)
