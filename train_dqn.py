from tqdm import tqdm
import gym
import env
from agents.dqn import DqnAgent
import torch
import numpy as np

def train(n_agents=4, n_turns=10, seed=42):
    max_move = 3
    n_collectables = n_agents * max_move * n_turns
    env = gym.make("BasketEnv-v0", n_agents=n_agents, n_collectables=n_collectables)
    episodes= 1000
    state, _ = env.reset(seed=0)
    n_actions = env.action_space.n
    loss = None
    input_size = 1000
    agent = DqnAgent(input_size, n_actions)
    epsilon_decay = 1e-2
    progress = tqdm(range(episodes))
    for episode in progress:
        epsilon = np.exp(- epsilon_decay * episode)
        state, info = env.reset(seed=seed + episode)
        done = False
        states = []
        actions = []
        rewards = []
        while not done:
            action = agent.move(env, state,epsilon=epsilon)
            state, reward, done, info = env.step(action)
            states.append(state)
            rewards.append(reward)
            actions.append(action)
        loss = agent.fit(states,actions,rewards)
        progress.set_description(f"Loss: {loss}", refresh=True)
    torch.save(agent.network.state_dict(), "dqn.pth")
    


if __name__ == "__main__":
    train()