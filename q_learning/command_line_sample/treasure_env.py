import os
import sys
import time

ACTIONS = ['left', 'right']     # available actions
FRESH_TIME = 0.3    # fresh time for one move

class FindTreasure(object):
    def __init__(self, dst=5):
        super(FindTreasure, self).__init__()
        self.position = 0
        self.dst = dst
        self.step_counter = 0

    def reset(self):
        self.position = 0
        self.step_counter = 0
        return self.position

    def step(self, action):
        if action == 'right':    # move right
            if self.position == self.dst - 1:   # terminate
                self.position = self.dst
                reward = 1
                done = True
            else:
                self.position = self.position + 1
                reward = 0
                done = False
        else:   # move left
            done = False
            reward = 0
            if self.position == 0:
                pass  # reach the wall
            else:
                self.position -= 1
        self.step_counter += 1
        return self.position, reward, done

    def render(self):
        # This is how environment be updated
        env_list = ['-']*self.dst + ['*']   # '---------T' our environment
        if self.position == self.dst:
            interaction = 'total_steps = %s' % (self.step_counter)
            print('\r{}'.format(interaction), end='')
            time.sleep(2)
            # print('\r                                ', end='')
            print('')
        else:
            env_list[self.position] = 'o'
            interaction = ''.join(env_list)
            print('\r{}'.format(interaction), end='')
            time.sleep(FRESH_TIME)
