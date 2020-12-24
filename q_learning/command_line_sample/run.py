"""
A simple example for Reinforcement Learning using table lookup Q-learning method.
An agent "o" is on the left of a 1 dimensional world, the treasure is on the rightmost location.
Run this program and to see how the agent will improve its strategy of finding the treasure.

View more on my tutorial page: https://morvanzhou.github.io/tutorials/
"""

from treasure_env import FindTreasure
import numpy as np
import pandas as pd


np.random.seed(2)  # reproducible


N_STATES = 6   # the length of the 1 dimensional world
ACTIONS = ['left', 'right']     # available actions
EPSILON = 0.9   # greedy police
ALPHA = 0.1     # learning rate
GAMMA = 0.9    # discount factor
MAX_EPISODES = 13   # maximum episodes
FRESH_TIME = 0.3    # fresh time for one move


def build_q_table(n_states, actions):
    table = pd.DataFrame(
        np.zeros((n_states, len(actions))),     # q_table initial values
        columns=actions,    # actions's name
    )
    # print(table)    # show table
    return table


def choose_action(state, q_table):
    # This is how to choose an action
    state_actions = q_table.iloc[state, :]
    if (np.random.uniform() > EPSILON) or ((state_actions == 0).all()):  # act non-greedy or state-action have no value
        action_name = np.random.choice(ACTIONS)
    else:   # act greedy
        action_name = state_actions.idxmax()    # replace argmax to idxmax as argmax means a different function in newer version of pandas
    return action_name


def update(q_table):
    env = FindTreasure(5)
    
    for episode in range(MAX_EPISODES):
        state = env.reset()
        env.render()

        done = False
        while not done:
            # choose action based on observation
            action = choose_action(state, q_table)

            # take action, get observation and reward
            new_state, reward, done = env.step(action)
            env.render()

            # learn from the transition
            q_predict = q_table.loc[state, action]

            if (not done):
                q_target = reward + GAMMA * q_table.iloc[new_state, :].max()
            else:
                q_target = reward

            q_table.loc[state, action] += ALPHA * (q_target - q_predict)  # update
            
            # move to next state
            state = new_state 

    return q_table


if __name__ == "__main__":
    q_table = build_q_table(N_STATES, ACTIONS)

    update(q_table)

    print('\r\nQ-table:\n')
    print(q_table)
