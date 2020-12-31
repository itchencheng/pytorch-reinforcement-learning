"""
A simple example for Reinforcement Learning using table lookup Q-learning method.
An agent "o" is on the left of a 1 dimensional world, the treasure is on the rightmost location.
Run this program and to see how the agent will improve its strategy of finding the treasure.

View more on my tutorial page: https://morvanzhou.github.io/tutorials/
"""

import treasure_env
from treasure_env import FindTreasure
import sys
sys.path.append('../../core/q_learning/')
from q_table_learning import QTableLearning
import numpy as np
import pandas as pd


np.random.seed(2)  # reproducible


MAX_EPISODES = 13   # maximum episodes
EPSILON = 0.9   # greedy police
ALPHA = 0.1     # learning rate
GAMMA = 0.9    # discount factor

def main():
    N_STATES = 7

    rl = QTableLearning(treasure_env.ACTIONS,
        learning_rate=ALPHA, reward_decay=GAMMA, e_greedy=EPSILON)
    env = FindTreasure(N_STATES)
    
    for episode in range(MAX_EPISODES):
        print("episode=%d" %(episode))

        state = env.reset()
        env.render()

        done = False
        while not done:
            # choose action based on observation
            action = rl.choose_action(state)

            # take action, get observation and reward
            new_state, reward, done = env.step(action)
            env.render()

            rl.learn(state, action, reward, new_state, done)

            # move to next state
            state = new_state 

    rl.show()


if __name__ == "__main__":
    main()