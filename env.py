from typing import Tuple
import gym
from gym import Env, spaces
import numpy as np
from copy import deepcopy

class BasketEnv(Env):

    def __init__(self, 
        n_agents:int, 
        n_collectables:int, 
        window_size=500,
        *args, **kwargs):
            super().__init__(*args, **kwargs)
            assert n_agents < n_collectables, "Must have more agents than collectables."
            self.window_size = window_size
            self.n_agents = n_agents
            self.n_collectables = n_collectables
            self.reset()

            self.observation_space = spaces.Dict(
                {
                    "agents": spaces.MultiDiscrete(self.n_agents),
                    "collectables": spaces.MultiBinary(self.n_collectables)
                }
            )
            # 1, 2 or 3
            self.action_space = spaces.Discrete(3)
    
    def _get_obs(self) -> dict:
        return {
            "agents": deepcopy(self.agents),
            "collectables": deepcopy(self.collectables)
        }

    def _get_info(self) -> dict:
        return {
            "next_turn": self.agents[0]
        }

    def step(self, action:int) -> Tuple[dict, int,bool,dict]:
        """
        Actions: [0,2]
        Reward: 1 for surviving, n_collectables for winning, 0 for losing.
        """
        
        action +=1
        agent = self.agents.pop(0)
        collectable = all(self.collectables.pop(0) for _ in range(action))
        reward = collectable if len(self.agents) > 1 else self.n_collectables
        if collectable == 1:
            self.agents.append(agent)
        
        done = len(self.agents) < 2
        obs = self._get_obs()
        info = self._get_info() | {"agent_died": collectable == 0}

        return obs, reward, done, info

    def reset(self, seed=123) -> Tuple[dict,dict]:
        rand = np.random.RandomState(seed)
        agents = np.arange(self.n_agents)
        rand.shuffle(agents)
        self.agents = list(agents)
        bad_collectables = rand.choice(self.n_collectables,self.n_agents - 1, replace=False)
        collectables = np.ones((self.n_collectables,))
        collectables[bad_collectables] = 0
        self.collectables = list(collectables)
        return self._get_obs(), self._get_info()

gym.envs.registration.register(
    id='BasketEnv-v0',
    entry_point='env:BasketEnv',
)

if __name__ == "__main__":
    env = gym.make('BasketEnv-v0', n_agents=1, n_collectables=20)
    env.reset()
    obs, *rest = env.step(0)
    print(obs)

