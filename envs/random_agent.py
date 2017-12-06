
class RandomAgent(object):
    def __init__(self, env):
        self.action_space = env.action_space

    def act(self, state):
        while True:
            target = self.action_space.sample()
            if state[1][target] == 0:
                return [state[0], target]
