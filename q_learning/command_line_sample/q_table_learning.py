
import numpy as np
import pandas as pd

class QTableLearning(object):
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
        super(QTableLearning, self).__init__()
        self.q_table = pd.DataFrame(columns=actions, dtype=np.float64)
        self.actions = actions
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
    

    def choose_action(self, state):
        self.check_state_exist(state)
        # action selection
        if np.random.uniform() < self.epsilon:
            # choose best action
            state_action = self.q_table.loc[state, :]
            # some actions may have the same value, randomly choose on in these actions
            action = np.random.choice(state_action[state_action == np.max(state_action)].index)
        else:
            # choose random action
            action = np.random.choice(self.actions)
        return action

    def learn(self, state, action, reward, new_state, done):
        self.check_state_exist(new_state)
        # learn from the transition
        q_predict = self.q_table.loc[state, action]
        if (not done):
            q_target = reward + self.gamma * self.q_table.loc[new_state, :].max()
        else:
            q_target = reward
        self.q_table.loc[state, action] += self.lr * (q_target - q_predict)  # update

    def check_state_exist(self, state):
        if state not in self.q_table.index:
            # append new state to q table
            self.q_table = self.q_table.append(
                pd.Series(
                    [0]*len(self.actions),
                    index=self.q_table.columns,
                    name=state,
                )
            )

    def load_model_from_file(self, csv_path):
        self.q_table = pd.read_csv(csv_path, index_col=0)
        self.q_table.columns = self.actions

    def save_model_to_file(self, csv_path):
        self.q_table.to_csv(csv_path)

    def show(self):
        print(self.q_table)
