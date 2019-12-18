import argparse
import os
import sys

print(sys.executable)

# Add gym_duckietown to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'gym-duckietown'))

from golem_driving.agents import get_agent
from golem_driving.envs import add_env_args, get_env_from_args
from golem_driving.run.show import show


modes = {
    'show': show
}


def run(mode, env, agent):
    if mode not in modes:
        print(f'Invalid mode: {mode}')
    else:
        modes[mode](env, agent)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', required=True)
    parser.add_argument('--agent', required=True)
    add_env_args(parser)
    args = parser.parse_args()

    with get_env_from_args(args, discrete=False) as env:
        try:
            agent = get_agent(args.agent)
        except KeyError:
            print(f'Invalid agent: {agent}')
            exit(-1)

        run(mode=args.mode, env=env, agent=agent)
