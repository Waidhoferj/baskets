import env
from agents.dqn import DqnAgent
from agents.random import RandomAgent
from agents.tree_search import TreeSearchAgent
from agents.dqn import DqnAgent
from tqdm import tqdm
import gym
import numpy as np

seed = 87

def train_agents():
    max_move = 3
    n_agents = 3
    n_turns = 5
    n_collectables = n_agents * max_move * n_turns
    episodes = 100

    rand_agent = RandomAgent(seed=seed)
    ts_agent = TreeSearchAgent()
    dqn = DqnAgent(1000, 3, seed=seed)
    dqn.load_weights("weights/dqn.pth")

    
    id_to_agent = [
        rand_agent,
        ts_agent,
        dqn
    ]
    assert len(id_to_agent) == n_agents, "Missing or overfilled agents."
    win_counts = [0] * len(id_to_agent)
    
    
    env = gym.make("BasketEnv-v0", n_agents=n_agents, n_collectables=n_collectables)

    for episode in tqdm(range(episodes)):
        state, info = env.reset(seed=seed + episode)
        done = False
        
        while not done:
            acting_agent = id_to_agent[info["next_turn"]]
            action = acting_agent.move(env, state)
            state, reward, done, info = env.step(action)
        winner = env.agents[0]
        win_counts[winner] += 1
    
    for place, i in enumerate(reversed(np.argsort(win_counts))):
        print(f"{place+1}.\t{id_to_agent[i]}:\t{win_counts[i]} wins")
    




    


if __name__ == "__main__":
    train_agents()