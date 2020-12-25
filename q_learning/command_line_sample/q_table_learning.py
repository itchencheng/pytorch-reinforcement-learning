
import numpy as np
import pandas as pd

EPSILON = 0.9   # greedy police
ALPHA = 0.1     # learning rate
GAMMA = 0.9    # discount factor

class QTableLearning(object):
    def __init__(self, n_states, actions):
        super(QTableLearning, self).__init__()
        self.table = pd.DataFrame(
            np.zeros((n_states, len(actions))), # q_table initial values
            columns=actions, # actions's name
        )
        self.actions = actions

    def choose_action(self, state):
        # This is how to choose an action
        state_actions = self.table.iloc[state, :]
        if (np.random.uniform() > EPSILON) or ((state_actions == 0).all()):  # act non-greedy or state-action have no value
            action_name = np.random.choice(self.actions)
        else:   # act greedy
            action_name = state_actions.idxmax()    # replace argmax to idxmax as argmax means a different function in newer version of pandas
        return action_name

    def learn(self, state, action, reward, new_state, done):
        # learn from the transition
        q_predict = self.table.loc[state, action]
        if (not done):
            q_target = reward + GAMMA * self.table.iloc[new_state, :].max()
        else:
            q_target = reward
        self.table.loc[state, action] += ALPHA * (q_target - q_predict)  # update

    def show(self):
        print(self.table)
