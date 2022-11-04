import numpy as np
import torch.nn as nn
import torch
from typing import List, Tuple


device = "mps"
epsilon_decay = 1e-6


class DqnAgent():

    def __init__(self, state_size:int,num_actions: int, seed=42):
        self.state_size = state_size
        self.input_size = state_size * 2
        self.num_actions = num_actions
        self.state_embedding = StateEmbedding(state_size)
        self.rand = np.random.RandomState(seed)
        network = nn.Sequential(
            nn.Linear(self.input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, self.num_actions)
        )
        self.network = network.to(device)
        self.memory = ReplayBuffer()
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=1e-4)

    def __repr__(self):
        return f"DqnAgent(state_size={self.state_size}, num_actions={self.num_actions})"


    def move(self,env, state, epsilon=0.0):
        features = self.state_embedding(state)
        features = torch.from_numpy(features).float().to(device)
        if self.rand.uniform(0, 1) < epsilon:
            return self.rand.choice(self.num_actions)
        else:
            with torch.no_grad():
                preds = self.network(features)
                action = torch.argmax(preds)
                return action.item()
    
    def fit(self,states:List[dict], actions: List[int], rewards: List[int], gamma=0.99):
        states = [self.state_embedding(s) for s in states]
        self.memory.append_episode(states, actions, rewards, gamma)
        states, actions, rewards, utilities = self.memory.sample(128)
        states = torch.from_numpy(states).float().to(device)
        utilities = torch.from_numpy(utilities).float().to(device)
        all_action_preds = self.network(states)
        action_indices = torch.from_numpy(np.expand_dims(actions, axis=-1)).type(torch.int64).to(device)
        predictions = torch.gather(all_action_preds, 1, action_indices)
        predictions = predictions[:, 0]
        
        loss = nn.functional.mse_loss(predictions, utilities)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def load_weights(self,path:str):
        self.network.load_state_dict(torch.load(path))




        


class ReplayBuffer:
    def __init__(self, seed=42):
        self.rand = np.random.RandomState(seed)
        self._states = []
        self._actions = np.array([])
        self._rewards = np.array([])
        self._utilities = np.array([])

    def __getitem__(self, index:int) -> Tuple[np.ndarray, int, int, float]:
        return self._states[index], self._actions[index], self._rewards[index], self._utilities[index] 
    
    def __len__(self) -> int:
        return len(self._states)
    
    def append_episode(self, states, actions,rewards, gamma=0.99):
        y_next = 0
        reward_scaler =  1.0 / float(len(states) *2)
        utilities = []
        for reward in reversed(rewards):
            y = reward + gamma * y_next
            y_next = y
            utilities.append(float(y) * reward_scaler)
        utilities = utilities[::-1]
        self._utilities = np.concatenate([self._utilities, utilities])
        self._states = np.concatenate([self._states, states]) if len(self._states) != 0 else np.array(states)
        self._actions = np.concatenate([self._actions, actions])
        self._rewards = np.concatenate([self._rewards, rewards])

    def sample(self, count=64) -> List[Tuple[np.ndarray, int, int, float]]:
        indices = self.rand.choice(len(self), count)
        return self._states[indices], self._actions[indices], self._rewards[indices], self._utilities[indices]


class StateEmbedding:

    def __init__(self, state_size:int):
        self.state_size = state_size


    def __call__(self, state:dict) -> np.ndarray:
        collectables = np.array(state["collectables"]) == 1.0
        end = min(self.state_size, len(collectables))
        collectables = collectables[:end]
        fruit = np.zeros((self.state_size,))
        bees = np.zeros((self.state_size,))
        fruit[:end] = collectables
        bees[:end] = np.invert(collectables)
        vector = np.concatenate([fruit, bees])
        return vector