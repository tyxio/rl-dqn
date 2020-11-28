# 
# https://towardsdatascience.com/tf-agents-tutorial-a63399218309
# https://github.com/sachag678/Reinforcement_learning/blob/master/tf-agents-example/gridworld.py
#
# Hands-on Machine learning with Scikit-Learn, Keras & Tensorflow 2nd edition, Aurelien Geron, chapter 18
# (https://www.amazon.fr/Hands-Machine-Learning-Scikit-learn-Tensorflow/dp/1492032646/ref=sr_1_1)

import abc
import numpy as np
import tensorflow as tf

# py -m pip install --upgrade tf-agents
from tf_agents.environments import py_environment
from tf_agents.environments import tf_environment
from tf_agents.environments import tf_py_environment
from tf_agents.environments import utils
from tf_agents.specs import array_spec
from tf_agents.trajectories  import time_step as ts
from tf_agents.environments import wrappers


# let’s start creating the custom Grid World environment which requires 
# sub-classing the PyEnvironment class
class GridWorldEnv(py_environment.PyEnvironment):

    def __init__(self):
        self._grid_size = 5
        self.inital_state =  [0, 0, self._grid_size, self._grid_size]

        # The action is a single integer ranging from 0 → 3
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.int32, minimum=0, maximum=3, name='action'
        )
        # The observation is 4 integers with max and min being 0 and _grid-size respectively
        # The first two integers refer to the row and column of the player, and the last
        #  two integers refer to the row and column of the win square.
        gs = self._grid_size
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(4,), dtype=np.int32, minimum=[0,0,0,0], maximum=[gs, gs, gs, gs], name='observation'
        )

        # initial state at (0,0) with the finish cell at the top rght corner
        self._state = [0, 0, gs, gs]
        self._episode_ended = False

    def observation_spec(self):
        return self._observation_spec

    def action_spec(self):
        return self._action_spec

    # The first two integers refer to the row and column of the player, and 
    #  the last two integers refer to the row and column of the win square.
    def _reset(self):
        self._state = self.inital_state
        self._episode_ended = False
        return ts.restart(np.array(self._state, dtype = np.int32))

    # The _step function handles state transition by taking an action and 
    #  applying it to the current state to get a new state.
    def _step(self, action):

        if (self._episode_ended):
            return self.reset()
        
        self.move(action)

        if self.check_game_over():
            self._episode_ended = True

        if self._episode_ended:
            if self.check_game_over():
                reward = 100
            else:
                reward = 0
            return ts.termination(
                np.array(self._state, dtype = np.int32), reward)
        else:
            return ts.transition(
                np.array(self._state, dtype = np.int32), reward = 0, discount = 0.9)
    
    # The move function is used to perform bounds checking and then 
    #  update the state based on an action.
    def move(self, action):
        row, col = self._state[0], self._state[1]
        
        if (action == 0): # down
            if (row - 1 >= 0):
                self._state[0] -= 1
        
        if (action == 1): # up
            if (row + 1 < self._grid_size + 1):
                self._state[0] += 1

        if (action == 2): # left
            if (col - 1 >= 0):
                self._state[1] -= 1

        if (action == 3): # right
            if (col + 1 < self._grid_size + 1):
                self._state[1] += 1


    # The game_over function checks to see if the game is over by comparing 
    #  the rows and columns of the player and the win square
    def check_game_over(self):
        return self._state[0] == self._state[2] and self._state[1] == self._state[3]

if __name__ == '__main__':
    env = GridWorldEnv()
    utils.validate_py_environment(env, episodes=5)

    tl_env = wrappers.TimeLimit(env, duration=50)

    time_step = tl_env.reset()
    print(time_step)
    rewards = time_step.reward

    for i in range(100):
        action = np.random.choice([0,1,2,3])
        time_step = tl_env.step(action)
        print(time_step)
        rewards += time_step.reward

    print(rewards)