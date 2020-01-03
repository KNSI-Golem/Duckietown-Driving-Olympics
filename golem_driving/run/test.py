import numpy as np


def test(env, agent, config):
    reward_list = []
    steps_list = []

    for _ in range(config.episodes):
        done = False
        total_award = 0
        total_steps = 0
        obs = env.reset()

        while not done:
            a = agent.act(obs)
            obs, reward, done, _ = env.step(a)
            total_award += reward
            total_steps += 1

        reward_list.append(total_award)
        steps_list.append(total_steps)

    print("Result after {} episodes:".format(config.episodes))
    print("Average reward per episode: {}".format(np.mean(reward_list)))
    print("Average steps per episode: {}".format(np.mean(steps_list)))
