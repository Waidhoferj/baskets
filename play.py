import env
from agents.dqn import DqnAgent
from agents.random import RandomAgent
from agents.tree_search import TreeSearchAgent
from agents.dqn import DqnAgent
from tqdm import tqdm
import gym
import numpy as np
from random import randint



def input_move() -> int:
    move = None
    choices = [1,2,3]
    message = f"Choose amount to collect: {choices}: "
    while True:
        move = input(message)
        try:
            move = int(move) 
            if move in choices: return move - 1
        except:
            continue

def display_game(state:dict, player_id:int):
    display = " ".join(["😎" if s == player_id else "🤖" for s in reversed(state["agents"])] + ["|"] + ["🍑" if s == 1.0 else "🐝" for s in state["collectables"]])
    print(display)
    


def play(n_collectables=20):
    ts_agent = TreeSearchAgent()
    dqn = DqnAgent(1000, 3, seed=randint(0,1024))
    rand = RandomAgent(seed=randint(0,1024))
    dqn.load_weights("dqn.pth")
    id_to_agent = [
        ts_agent,
        dqn,
        rand
    ]
    player_id = len(id_to_agent)
    env = gym.make("BasketEnv-v0", n_agents=len(id_to_agent) + 1, n_collectables=n_collectables)
    state, info = env.reset(seed=randint(0,1024))
    display_game(state, player_id)
    done = False
    while not done:
        players_turn = info["next_turn"] == player_id
        if players_turn:
            action = input_move()
        else:
            acting_agent = id_to_agent[info["next_turn"]]
            action = acting_agent.move(env, state)
        state, _, done, info = env.step(action)
        display_game(state, player_id)
    winner = env.agents[0]
    if winner == len(id_to_agent):
        print("You won!")
    else:
        print(f"{id_to_agent[winner]} won.")
    




    


if __name__ == "__main__":
    play()