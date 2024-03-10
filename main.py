import random
import numpy as np

from agent import Agent
from environment import Game

def epsilon_greedy_policy(env, epsilon):
    if random.uniform(0, 1) > epsilon:
        return np.argmax(env.q_table[env.get_state()])
    return random.randint(0, 3)

def train(
    env, 
    n_episodes, 
    min_epsilon, max_epsilon, 
    learning_rate, decay_rate, gamma) -> np.ndarray:
    """_summary_

    Args:
        env (_type_): _description_
        n_episodes (_type_): _description_
        min_epsilon (_type_): _description_
        max_epsilon (_type_): _description_
        learning_rate (_type_): _description_
        decay_rate (_type_): _description_
        gamma (_type_): _description_

    Returns:
        np.ndarray: _description_
    """
    for episode in range(n_episodes):
        # decay
        epsilon = min_epsilon + (max_epsilon - min_epsilon)*np.exp(-decay_rate*episode)
        state = env.reset()
        done = False
        
        while not done:
            # get action
            action = epsilon_greedy_policy(env, epsilon)
            # get new state, reward and possible stop
            new_state, reward, done = env.step(action)
            # update QTable
            print(env.q_table)
            print(state, action)
            env.q_table[state][action] = (
                env.q_table[state][action] 
                + learning_rate 
                * (reward + gamma * np.max(env.q_table[new_state]) - env.q_table[state][action])
            )
            if done:
                break
    return env.q_table


if __name__ == '__main__':
    env = Game()
    n_episodes = 1_000 
    min_epsilon = 0.05
    max_epsilon = 0.95
    learning_rate = 1e-6
    decay_rate = 0.01
    gamma = 0.9

    QTable = train(
        env, n_episodes, min_epsilon, max_epsilon, learning_rate, decay_rate, gamma
    )