import environment


if __name__ == "__main__":
    env = environment.TicTacToeEnv()

    state_size = env.observation_space.shape
    action_size = env.action_space.shape
    print(state_size, action_size)
