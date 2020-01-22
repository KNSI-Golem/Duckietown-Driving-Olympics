import gym
from argparse import ArgumentParser
from typing import Callable

from gym_duckietown.envs.duckietown_env import DuckietownEnv
from gym_duckietown.simulator import Simulator
from gym_duckietown.wrappers import DiscreteWrapper


def add_env_args(parser: ArgumentParser) -> type(None):
    parser.add_argument('--env-name', default=None)
    parser.add_argument('--map-name', default='loop_empty')
    parser.add_argument('--distortion', default=False, action='store_true')
    parser.add_argument('--draw-curve', action='store_true', help='draw the lane following curve')
    parser.add_argument('--draw-bbox', action='store_true', help='draw collision detection bounding boxes')
    parser.add_argument('--domain-rand', action='store_true', help='enable domain randomization')
    parser.add_argument('--frame-skip', default=1, type=int, help='number of frames to skip')
    parser.add_argument('--frame-rate', default=None, help='number of frames per second')


def get_env_from_args(args, discrete: bool=False) -> Callable[[], Simulator]:
    def builder():
        if args.env_name is None:
            env = DuckietownEnv(
                map_name=args.map_name,
                draw_curve=args.draw_curve,
                draw_bbox=args.draw_bbox,
                domain_rand=args.domain_rand,
                frame_skip=args.frame_skip,
                distortion=args.distortion,
            )
        else:
            env = gym.make(args.env_name)

        return DiscreteWrapper(env) if discrete else env

    return builder()
