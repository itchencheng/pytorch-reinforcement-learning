import gym
import grid_game_env
import numpy as np

import os
import sys
sys.path.append('../../core/q_learning/')
import q_table_learning

env = gym.make("CliffWalking-v0")  # 0 up, 1 right, 2 down, 3 left
env = grid_game_env.CliffWalkingWapper(env)
model_path = 'model/cliff_walking.csv'

'''
env = gym.make("FrozenLake-v0", is_slippery=False)
env = grid_game_env.FrozenLakeWapper(env)
model_path = 'model/frozen_lake.csv'
'''

'''
env = gym.make("FrozenLake-v0", is_slippery=True)
env = grid_game_env.FrozenLakeWapper(env)
model_path = 'model/frozen_lake_slippery.csv'
'''

def main(max_episodes):
    ACTIONS = [0, 1, 2, 3]
    rl = q_table_learning.QTableLearning(ACTIONS,
        learning_rate=0.1, reward_decay=0.9, e_greedy=0.9)
    rl.load_model_from_file(model_path)

    for episode in range(max_episodes):
        state = env.reset()

        env.render()

        done = False
        step = 0
        while not done:
            action = rl.choose_action(state)
           
            new_state, reward, done, x = env.step(action)
            env.render()
            
            rl.learn(state, action, reward, new_state, done)
            
            state = new_state 
            
            step += 1

        print(("done", episode, step))

    rl.show()
    rl.save_model_to_file(model_path)

    print("finished")

if __name__ == "__main__":
    max_episodes = int(sys.argv[1])
    main(max_episodes)