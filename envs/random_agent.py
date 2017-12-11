
class RandomAgent(object):
    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, state):
        while True:
            target = self.action_space.sample()
            if state[1][target] == 0:
                return [state[0], target]
