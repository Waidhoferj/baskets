from gym import Env
import numpy as np


class RandomAgent:

    def __init__(self, seed=123):
        self.seed=seed
        self.rand = np.random.RandomState(seed)

    def __repr__(self):
        return f"RandomAgent(seed={self.seed})"

    def move(self,env: Env, state:dict) -> int:
        action = self.rand.randint(env.action_space.n)
        return action



