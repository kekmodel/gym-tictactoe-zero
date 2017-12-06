import logging
import sys
import gym
import envs
from gym import wrappers


class PPOAgent:
    def __init__(self, observation_space, action_space):
        self.observation_space = observation_space
        self.action_space = action_space
        self.action_n = action_space.n

    def get_action(self):
        return


if __name__ == "__main__":
    env = gym.make('TicTacToe-v0')
    outdir = '/tmp/ppo-agent-results'
    env = wrappers.Monitor(env, directory=outdir, force=True)
    state_space = env.observation_space
    action_space = env.action_space
    agent_A = PPOAgent(state_space, action_space)


'''
    for i in range(100):
        action = agent_A.get_action
        state, reward, done, info = env.step(action)
        print(reward)
        print(state)
        env.render()
        if done:
            env.reset()
'''
