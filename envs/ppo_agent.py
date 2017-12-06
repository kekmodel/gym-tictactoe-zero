import logging
import gym
import envs
from random_agent import RandomAgent
# from gym import wrappers


class PPOAgent(object):
    def __init__(self, env):
        self.action_space = env.action_space


if __name__ == "__main__":
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    env = gym.make('TicTacToe-v0')
    agent_A = RandomAgent(env)

    # outdir = 'tmp/ppo-agent-results'
    # env = wrappers.Monitor(env, directory=outdir, force=True)

    episode_count = 10000
    reward = 0
    done = False
    result = {1: 0, 0: 0, -1: 0}
    my_mark = env.observation_space[0]

    for i in range(episode_count):
        state = env.reset()
        env.player = my_mark.sample()
        print('-' * 15, '\nepisode: %d' % (i + 1))
        while True:
            action = agent_A.act(state)
            state, reward, done, info = env.step(action)
            if done:
                result[reward] += 1
                break
    print(result)

'''
    state_space = env.observation_space
    action_space = env.action_space
    agent_A = PPOAgent(state_space, action_space)
    env.reset()
    for i in range(100):
        action = agent_A.get_action
        state, reward, done, info = env.step(action)
        print(reward)
        print(state)
        env.render()
        if done:
            env.reset()
'''
