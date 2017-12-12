import gym
import envs
import time
# from random_agent import RandomAgent


class MyAgent(object):
    def __init__(self, action_space):
        self.action_space = action_space
        self.action_n = action_space.n

    def act(self, state):
        while True:
            target = self.action_space.sample()
            if state[1][target] == 0:
                action = [state[0], target]
                return action


if __name__ == "__main__":

    env = gym.make('TicTacToe-v0')
    env.seed(2017)

    agent = MyAgent(env.action_space)
    my_mark = env.observation_space[0]

    episode_count = 10
    reward = 0
    done = False
    result = {1: 0, 0: 0, -1: 0}
    mark_dict = {0: 'O', 1: 'X'}

    for i in range(episode_count):
        state = env.reset()
        env.player = my_mark.sample()
        print('-' * 15, '\nepisode: %d' % (i + 1))
        print('My Mark: %s' % mark_dict[env.player])
        while True:
            action = agent.act(state)
            state, reward, done, _ = env.step(action)
            if done:
                result[reward] += 1
                print('timesteps: %d' % (state[2] + 1))
                break
    env.close()
    winrate = result[0] / episode_count * 100
    print('-' * 15, '\nWin: %d Lose: %d Draw: %d Winrate: %0.1f%%' %
          (result[1], result[-1], result[0], winrate))

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
