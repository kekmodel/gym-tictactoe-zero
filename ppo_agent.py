import environment


class PPOAgent:
    def __init__(self, state_size, action_size):
        pass

    def get_action(self):
        return


if __name__ == "__main__":
    env = environment.TicTacToeEnv()
    state_size = env.observation_space.shape
    action_size = env.action_space.shape
    env.reset()

    agent_A = PPOAgent(state_size, action_size)

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
