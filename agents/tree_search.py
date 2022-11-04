from copy import deepcopy
from gym import Env
import numpy as np

class TreeSearchAgent:
    """
    Finds the path that is least likely to end in hitting the bad collectable through DFS.
    """

    def __repr__(self):
        return "TreeSearchAgent"

    def move(self, env: Env, state:dict) -> int:
        rollout_depth = len(state["agents"])
        my_id = env.agents[0]
        best_move = np.argmax(self._rollout(env,state, rollout_depth, my_id))
        return best_move

    def _rollout(self,env: Env, state:dict, rollout_depth:int, my_id:int) -> list:
        """
        Looks ahead one cycle and makes the move that heads towards the best average reward.
        """
        my_move = state["agents"][0] == my_id
        expected_reward = []
        for action in range(env.action_space.n):
            possible_future = deepcopy(env)
            state,reward, done, info = possible_future.step(action)
            has_died = my_move and info["agent_died"]
            if not done and not has_died and rollout_depth > 1:
                reward += np.mean(self._rollout(possible_future, state,rollout_depth-1, my_id))
            expected_reward.append(reward if my_move else 0)
        return expected_reward
    

    




    